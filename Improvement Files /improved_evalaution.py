from __future__ import print_function
import argparse, random
import torch, torchvision
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.utils.data.sampler as sp
import foolbox as fb
from foolbox.criteria import Misclassification, TargetedMisclassification
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from vgg import VGG
from resnet import ResNet50
import os
import logging

logging.basicConfig(
    filename='improved_evaluation.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def show_images(tensor, title="", nrow=8):
    """
    Display a grid of images from a B*C*H*W or C*H*W tensor.
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    grid = make_grid(tensor.cpu(), nrow=nrow, normalize=True, scale_each=True)
    npimg = grid.permute(1,2,0).numpy()
    plt.figure(figsize=(6,6))
    plt.imshow(npimg)
    plt.title(title)
    plt.axis('off')
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description='DaST/CIFAR-10 Attack Evaluation')
    parser.add_argument('--mode', choices=['baseline', 'dast', 'white'], required=True,
                        help='Which substitute to use: baseline=ResNet-50, dast=DaST-VGG13, white=target itself')
    parser.add_argument('--adv', choices=['FGSM','BIM','PGD','CW'], required=True,
                        help='Attack method')
    parser.add_argument('--targeted', action='store_true', help='Run targeted attack')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dast-model', type=str, default='saved_model5/netD_epoch_534.pth',help='Path to DaST substitute')
    parser.add_argument('--baseline-model', type=str, default='pretrained/resnet50_cifar10.pth',help='Path to ResNet-50 baseline')
    parser.add_argument('--target-model', type=str, default='Improvement Files/vgg_vgg16_final.pth',help='Path to VGG-16 target')
    parser.add_argument('--epsilon', type=float, default=0.031, help='L∞ perturbation budget for FGSM/BIM/PGD (e.g. 0.031)')
    parser.add_argument('--output-dir', type=str, default='outputs/synthetic',
                        help='Directory to save generated images')
    return parser.parse_args()


def build_attack(name, targeted):
    if name == 'FGSM':
        return fb.attacks.LinfFastGradientAttack()
    if name == 'BIM':
        return fb.attacks.LinfBasicIterativeAttack(abs_stepsize=2/255, steps=10, random_start=False)
    if name == 'PGD':
        return fb.attacks.LinfPGD(steps=20, abs_stepsize=2/255, random_start=True)
    if name == 'CW':
        return fb.attacks.L2CarliniWagnerAttack(binary_search_steps=9, steps=1000, stepsize=1e-2)
    raise ValueError(f"Unknown attack {name}")

# … inside test_adversary, before the loop or at top of file …
os.makedirs("outputs", exist_ok=True)

def test_adversary(sub_model, target_model, attack, args, testloader):
    sub_model.eval()
    target_model.eval()
    fmodel = fb.PyTorchModel(sub_model, bounds=(0,1))
    criterion_cls = TargetedMisclassification if args.targeted else Misclassification

    correct = 0.0
    total = 0.0
    total_L2_distance  = 0.0
    first= True 
    #for inputs, labels in testloader:
    for data in testloader:
        inputs,labels = data
        inputs, labels = inputs.to(args.device), labels.to(args.device)
        with torch.no_grad():
            pred = target_model(inputs)
            predicted= pred.argmax(dim=1)
        
        if args.targeted:
            # randomly choose the specific label of targeted attack
            targeted_labels = torch.randint(0, 10, (inputs.size(0),), device=args.device)
            # test the images which are not classified as the specific label
            # make sure it's different from orig
            targeted_labels = torch.where(targeted_labels == predicted, (targeted_labels + 1) % 10, targeted_labels)
            mask= predicted != targeted_labels

            if not mask.any():
               continue
            inputs, targeted_labels = inputs[mask], targeted_labels[mask]
            criterion = TargetedMisclassification(targeted_labels)
        else:
            # test the images which are classified correctly
            mask= predicted == labels
            if not mask.any():
                continue
                # print(total)
            inputs, labels = inputs[mask], labels[mask]
            criterion = Misclassification(labels)


        # —— craft adversarial examples & get their ℓ₂s ——
        adv_inputs_ori, clipped, is_adv = attack(fmodel, inputs, criterion, epsilons=args.epsilon)
        advs = adv_inputs_ori.to(args.device)
        #L2_distance = (torch.norm(adv_inputs_ori - inputs, p=2)).item()
        batch_l2 = torch.norm(
            (advs - inputs).view(advs.size(0), -1),
            dim=1, p=2
        )
        total_L2_distance += batch_l2.sum().item()
    
        # —— evaluate on target_model ——
        with torch.no_grad():
            adv_pred = target_model(advs).argmax(dim=1)
        if args.targeted:
            correct += float((adv_pred == labels).sum().item())
            total += labels.size(0)
        else:
            correct += float((adv_pred == labels).sum().item())
            total += labels.size(0)
        

        if first:
            # save 8×8 grid of the first 64 clean/adversarial
            save_image(
                inputs[:64],
                "outputs/clean_8x8.png",
                nrow=8, normalize=True
            )
            save_image(
                advs[:64],
                f"outputs/adv_{args.adv}_8x8.png",
                nrow=8, normalize=True
            )
            # also dump a high‐res matplotlib version (300dpi, 10"×10")
            grid = make_grid(advs[:64].cpu(), nrow=8, normalize=True)
            plt.figure(figsize=(10,10))
            plt.axis('off')
            plt.imshow(grid.permute(1,2,0))
            plt.close()
            first = False
        
    ASR = (100. * correct / total) if args.targeted else (100.0 - 100. * correct / total)
    L2D = total_L2_distance / total

    if args.targeted:
        msg = (f"{args.adv} Targeted   | ASR: {ASR:.2f}% (n={correct},m={total}) | L2: {L2D:.4f}")
    else:
        msg = (f"{args.adv} Untargeted | ASR: {ASR:6.2f}% (n={correct},m={total}) | L2: {L2D:.4f}")
    print(msg)
    logging.info(msg)
       


def main():
    args = parse_args()
    cudnn.benchmark = True
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size,
        sampler=sp.SubsetRandomSampler(list(range(len(testset)))),
        num_workers=2
    )

    # Load target (black-box) model
    target_model = VGG('VGG16').to(args.device)
    ck = torch.load('Improvement Files/vgg_vgg16_final.pth', map_location=args.device)
    target_model.load_state_dict(ck)
    target_model.eval()

    # Load substitute
    if args.mode == 'white':
        sub_model = target_model
    elif args.mode == 'baseline':
        sub_model = ResNet50().to(args.device)
        sub_model.load_state_dict(torch.load(args.baseline_model, map_location=args.device))
    elif args.mode == 'dast':
        sub_model = VGG('VGG13').to(args.device)
        sub_model.load_state_dict(torch.load(args.dast_model, map_location=args.device))
    else:
        raise ValueError(f"Unknown mode {args.mode}")
        
            
    # Build attack
    attack = build_attack(args.adv, args.targeted)

    # Evaluate
    test_adversary(sub_model, target_model, attack, args, testloader)
    
    # ─── Substitute model accuracy on real CIFAR-10 ───
    sub_model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(args.device), labels.to(args.device)
            outputs = sub_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100. * correct / total
    print(f"Substitute model accuracy: {acc:.2f}%")
    logging.info(f"Substitute model accuracy: {acc:.2f}%")
if __name__ == '__main__':
    main()


#This single script now:  
#- Loads CIFAR‑10 test set (all 10k samples).  
#- Supports `--mode` (baseline, dast, white), `--adv`, and `--targeted`.  
#- Evaluates both non-targeted and targeted success rates and L2 distances.  
