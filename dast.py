from __future__ import print_function
import argparse
import os
import math
import gc
import sys
import xlwt
import random
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.data.sampler as sp
import foolbox as fb
from foolbox.criteria import Misclassification
from vgg import VGG  # Make sure you have your VGG model defined properly

cudnn.benchmark = True
workbook = xlwt.Workbook(encoding='utf-8')
worksheet = workbook.add_sheet('imitation_network_sig')
nz = 128

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

sys.stdout = Logger('imitation_network_model.log', sys.stdout)

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=500)
parser.add_argument('--niter', type=int, default=2000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--cuda', default=True, action='store_true')
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--beta', type=float, default=0.1)
parser.add_argument('--G_type', type=int, default=1)
parser.add_argument('--save_folder', type=str, default='saved_model')
parser.add_argument('--original_model_path', type=str, required=True)
opt = parser.parse_args()
print(opt)
if opt.cuda and not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available")

device = torch.device("cuda:0" if opt.cuda else "cpu")

transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batchSize, sampler=sp.SubsetRandomSampler([i for i in range(6000, 8000)]), num_workers=opt.workers)

netD = VGG('VGG13').to(device)
netD = nn.DataParallel(netD)

original_net = VGG('VGG16').to(device)
original_net.load_state_dict(torch.load(opt.original_model_path))
original_net.eval()

fmodel = fb.PyTorchModel(original_net, bounds=(0.0, 1.0))
attack = fb.attacks.L2BasicIterativeAttack(abs_stepsize=0.01, steps=200, random_start=False)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Loss_max(nn.Module):
    def __init__(self):
        super(Loss_max, self).__init__()

    def forward(self, pred, truth, proba):
        criterion_1 = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        pred_prob = F.softmax(pred, dim=1)
        loss = criterion(pred, truth) + criterion_1(pred_prob, proba) * opt.beta
        final_loss = torch.exp(loss * -1)
        return final_loss

class pre_conv(nn.Module):
    def __init__(self, num_class):
        super(pre_conv, self).__init__()
        self.nf = 64
        if opt.G_type == 1:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif opt.G_type == 2:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),  # added

                # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 8),
                # nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),
                nn.ReLU(True),

                nn.Conv2d(self.nf, self.shape[0], 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.shape[0]),
                nn.ReLU(True),

                nn.Conv2d(self.shape[0], self.shape[0], 3, 1, 1, bias=False),
                # if self.shape[0] == 3:
                #     nn.Tanh()
                # else:
                nn.Sigmoid()
            )
    def forward(self, input):
        output = self.pre_conv(input)
        return output

pre_conv_block = []
for i in range (10):
    pre_conv_block.append(nn.DataParallel(pre_conv(10).cuda()))

class Generator(nn.Module):
    def __init__(self, num_class):
        super(Generator, self).__init__()
        self.nf = 64
        self.num_class = num_class
        if opt.G_type == 1:
            self.main = nn.Sequential(
                nn.Conv2d(self.nf * 2, self.nf * 4, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                # nn.Conv2d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 4),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 8, 3, 1, 0, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.LeakyReLU(0.2, inplace=True),

                # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 8),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf * 2, self.nf, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(self.nf, nc, 3, 1, 1, bias=False),
                nn.BatchNorm2d(nc),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(nc, nc, 3, 1, 1, bias=False),
                nn.Sigmoid()
            )
        elif opt.G_type == 2:
            self.main = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 2, self.nf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 4, self.nf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 4, self.nf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.ConvTranspose2d(self.nf * 8, self.nf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True),

                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 8),
                nn.ReLU(True)
            )
    def forward(self, input):
        output = self.main(input)
        return output

def chunks(arr, m):
    n = int(math.ceil(arr.size(0) / float(m)))
    return [arr[i:i + n] for i in range(0, arr.size(0), n)]

netG = Generator(10).cuda()
netG.apply(weights_init)
netG = nn.DataParallel(netG)

criterion = nn.CrossEntropyLoss()
criterion_max = Loss_max()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerD =  optim.SGD(netD.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
# optimizerG =  optim.SGD(netG.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
optimizer_block = []
for i in range(10):
    optimizer_block.append(optim.Adam(pre_conv_block[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))

with torch.no_grad():
    correct_netD = 0.0
    total = 0.0
    netD.eval()
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = original_net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_netD += (predicted == labels).sum()
    print('Accuracy of original model: %.2f %%' % (100. * correct_netD.float() / total))

################################################
# estimate the attack success rate of initial D:
################################################

correct_ghost = 0.0
total = 0.0
netD.eval()
for data in testloader:
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    adv_inputs, _, _ = attack(fmodel, inputs, Misclassification(labels), epsilons=1.5)
    with torch.no_grad():
        outputs = original_net(adv_inputs)
        _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct_ghost += (predicted == labels).sum()
print('Attack success rate: %.2f %%' % (100 - 100. * correct_adv.float() / total))
del inputs, labels, adv_inputs_ghost
torch.cuda.empty_cache()
gc.collect()

batch_num = 1000
best_accuracy = 0.0
best_att = 0.0
for epoch in range(opt.niter):
    netD.train()

    for ii in range(batch_num):
        netD.zero_grad()
        ############################
        # (1) Update D network:
        ###########################
        noise = torch.randn(opt.batchSize, nz, 1, 1, device=device).cuda()
        noise_chunk = chunks(noise, 10)
        for i in range(len(noise_chunk)):
            tmp_data = pre_conv_block[i](noise_chunk[i])
            gene_data = netG(tmp_data)
            # gene_data = netG(noise_chunk[i], i)
            label = torch.full((noise_chunk[i].size(0),), i).cuda()
            if i == 0:
                data = gene_data
                set_label = label
            else:
                data = torch.cat((data, gene_data), 0)
                set_label = torch.cat((set_label, label), 0)
        index = torch.randperm(set_label.size()[0])
        data = data[index]
        set_label = set_label[index]

        with torch.no_grad():
                outputs = original_net(data)
                _, labels = torch.max(outputs.data, 1)
                soft_labels = F.softmax(outputs, dim=1)

        outputs_D = netD(data.detach())
        prob_D = F.softmax(outputs_D, dim=1)
        errD_prob = F.mse_loss(prob_D, soft_labels)
        errD_fake = criterion(outputs_D, labels) + errD_prob * opt.beta
        errD_fake.backward()
        optimizerD.step()
        del outputs_D, errD_fake

        ############################
        # (2) Update G network:
        ###########################

        netG.zero_grad()
        for block in pre_conv_block:
            block.zero_grad()

        output_G = netD(data)
        loss_imitate = criterion_max(output_G, labels, soft_labels)
        loss_diversity = criterion(output_G, set_label.squeeze().long())
        errG = opt.alpha * loss_diversity + loss_imitate
        if loss_diversity.item() <= 0.1:
            opt.alpha = loss_diversity.item()
        errG.backward()
        optimizerG.step()
        for opt_block in optimizer_block:
            opt_block.step()

        if ii % 40 == 0:
            print('[%d/%d][%d/%d] D Loss: %.4f, D_prob Loss: %.4f, G Loss: %.4f, Imitation: %.4f, Diversity: %.4f' %
                  (epoch, opt.niter, ii, batch_num, errD_fake.item(), errD_prob.item(), errG.item(), loss_imitate.item(), loss_diversity.item()))
    ################################################
    # estimate the attack success rate of trained D:
    ################################################
    correct_ghost = 0.0
    total = 0.0
    netD.eval()
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        adv_inputs, _, _ = attack(fmodel, inputs, Misclassification(labels), epsilons=1.5)
        with torch.no_grad():
            outputs = original_net(adv_inputs)
            _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct_ghost += (predicted == labels).sum()

    print('Attack success rate: %.2f %%' % (100 - 100. * correct_ghost.float() / total))
    worksheet.write(epoch, 0, (correct_ghost.float() / total).item())

    if best_att < (total - correct_ghost):
        torch.save(netD.state_dict(), f'{opt.save_folder}/netD_epoch_{epoch}.pth')
        torch.save(netG.state_dict(), f'{opt.save_folder}/netG_epoch_{epoch}.pth')
        best_att = (total - correct_ghost)
        print('This is the best model')
    worksheet.write(epoch, 0, (correct_ghost.float() / total).item())
    del inputs, labels, adv_inputs_ghost
    torch.cuda.empty_cache()
    gc.collect()

    ################################################
    # evaluate the accuracy of trained D:
    ################################################
    with torch.no_grad():
        correct_netD = 0.0
        total = 0.0
        netD.eval()
        for data in testloader:
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            outputs = netD(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct_netD += (predicted == labels).sum()
        print('Accuracy of the network on netD: %.2f %%' %
                (100. * correct_netD.float() / total))
        if best_accuracy < correct_netD:
            torch.save(netD.state_dict(),
                       opt.save_folder + '/netD_epoch_%d.pth' % (epoch))
            torch.save(netG.state_dict(),
                       opt.save_folder + '/netG_epoch_%d.pth' % (epoch))
            best_accuracy = correct_netD
            print('This is the best model')
    worksheet.write(epoch, 1, (correct_netD.float() / total).item())
workbook.save('imitation_network_saved_azure.xls')
    
