
"""
 script to generate and save improved synthetic images from a trained DaST generator (CIFAR-10).
"""
import os
import argparse
import torch
import torch.nn as nn
from torchvision.utils import save_image , make_grid
import math
from PIL import Image

nz = 128  # default latent dimension if not overridden

# -----------------------------------------------------------------------------
# User-defined blocks (adapted from dast_cifar10_imp.py)
# -----------------------------------------------------------------------------

class pre_conv(nn.Module):
    def __init__(self, nz, G_type):
        super(pre_conv, self).__init__()
        self.nf = 64
        self.nz = nz
        if G_type == 1:
            self.pre_conv = nn.Sequential(
                nn.Conv2d(nz, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2), nn.ReLU(True),
                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2), nn.ReLU(True),
                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2), nn.ReLU(True),
                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2), nn.ReLU(True),
                nn.ConvTranspose2d(self.nf * 2, self.nf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2), nn.ReLU(True)
            )
        elif G_type == 2:
            # second architecture branch from original dast_cifar10_imp
            self.pre_conv = nn.Sequential(
                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, round((self.shape[0]-1) / 2), bias=False),
                nn.BatchNorm2d(self.nf * 8), nn.ReLU(True),
                nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1,round((self.shape[0]-1) / 2) , bias=False),
                nn.BatchNorm2d(self.nf * 8), nn.ReLU(True),
                nn.Conv2d(self.nf * 8, self.nf * 4, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 4), nn.ReLU(True),
                nn.Conv2d(self.nf * 4, self.nf * 2, 3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf * 2), nn.ReLU(True),
                nn.Conv2d(self.nf * 2, self.nf,   3, 1, 1, bias=False),
                nn.BatchNorm2d(self.nf),     nn.ReLU(True),
                nn.Conv2d(self.nf, self.shape[0], 3,   3, 1, 1, bias=False),
                nn.BatchNorm2d(3),(self.shape[0]),nn.ReLU(True),
                nn.Conv2d(3, 3, 3, 1, 1, bias=False), 
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unsupported G_type: {G_type}")

    def forward(self, x):
        return self.pre_conv(x)

class GeneratorCIFAR10(nn.Module):
    def __init__(self, G_type):
        super(GeneratorCIFAR10, self).__init__()
        self.nf = 64
        #self.num_class = num_class
        if G_type == 1:
            # input channels = 128 from PreConv
            self.main = nn.Sequential(
                nn.Conv2d(128, 256, 3, 1, 1, bias=False),          #64 32 32
                nn.BatchNorm2d(256),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                # nn.Conv2d(self.nf * 4, self.nf * 4, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 4),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(256, 512, 3, 1, 1, bias=False),        #32 32 32
                nn.BatchNorm2d(512),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                # nn.Conv2d(self.nf * 8, self.nf * 8, 3, 1, 1, bias=False),
                # nn.BatchNorm2d(self.nf * 8),
                # nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(512, 256, 3, 1, 1, bias=False),          #16 32 32
                nn.BatchNorm2d(256),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                nn.Conv2d(256, 128, 3, 1, 1, bias=False),         #8 32 32
                nn.BatchNorm2d(128),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                nn.Conv2d(128, 64, 3, 1, 1, bias=False),         #4 32 32
                nn.BatchNorm2d(64),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                nn.Conv2d(64, 3, 3, 1, 1, bias=False),     #2 32 32
                nn.BatchNorm2d(3),
                # nn.LeakyReLU(0.2, inplace=True),
                nn.ReLU(True),

                nn.Conv2d(3, 3, 3, 1, 1, bias=False),   #1 28 28--->3 32 32
               # nn.BatchNorm2d(3),#---------
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
        else:
            raise ValueError(f"Unsupported G_type: {G_type}")

    def forward(self, x):
        return self.main(x)

# -----------------------------------------------------------------------------
# Script to load a checkpoint and generate images
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Synthetic DaST Images")
    parser.add_argument('--generator-checkpoint', type=str, required=True,
                        help='Path to trained netG .pth file')
    parser.add_argument('--G-type', type=int, default=1,
                        help='Generator type (must match training)')
    parser.add_argument('--nz', type=int, default=128,
                        help='Latent vector dimension')
    parser.add_argument('--num-images', type=int, default=64,
                        help='Total synthetic images to generate')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Number of images per row in the output grid')
    parser.add_argument('--output-dir', type=str, default='outputs/synthetic_improved_dast_l',
                        help='Directory to save generated images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cpu or cuda')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Build per-class pre_conv blocks
    #pre_conv_block = [pre_conv(args.nz, args.G_type).to(device) for i in range(10)
    pre_conv_block = []
    for i in range (10):
    #pre_conv_block.append(nn.DataParallel(pre_conv(10).cuda()))
        pre_conv_block.append(pre_conv(nz, args.G_type).to(device))

    # Load generator
    netG = GeneratorCIFAR10(args.G_type).to(device)
    #netG = Generator_cifar10(10).cuda()
    ckpt = torch.load(args.generator_checkpoint, map_location=device)
    netG.load_state_dict(ckpt)
    netG.eval()
    
    # Sample z and classes
    z = torch.randn(args.num_images, args.nz, 1, 1, device=device)
    classes = torch.arange(10, device=device).repeat(args.num_images // 10 + 1)[:args.num_images]

    # Generate synthetic images
    synth_chunks = []
    for c in range(10):
        idxs = (classes == c).nonzero(as_tuple=True)[0]
        if idxs.numel() > 0:
            pc_in = pre_conv_block[c](z[idxs])
            synth_chunks.append(netG(pc_in))
    synth = torch.cat(synth_chunks, dim=0)

    #  Grab exactly 64 samples
    tosave = synth[:64]  # make sure you actually generated ≥64 images

    #  Make an 8×8 grid
    grid = make_grid(
        tosave,
        nrow=8,            # 8 images per row → 8 rows if you have 64 imgs
        padding=2,
        normalize=True,
        scale_each=True
    )

    #  Convert to numpy/PIL so we can up‐size it
    ndarr = (grid.mul(255)
                .add_(0.5)
                .clamp_(0,255)
                .permute(1,2,0)  # C×H×W → H×W×C
                .to("cpu", torch.uint8)
                .numpy())
    img = Image.fromarray(ndarr)

    #  (Optional) Upscale for visibility
    #    here we blow it up to 512×512 with nearest‐neighbor
    img = img.resize((512,512), Image.NEAREST)

    #  Save
    out_path = os.path.join(args.output_dir, "synthetic_8x8_grid.png")
    img.save(out_path)
    print(f"Saved 8×8 synthetic grid → {out_path}")
#  Save up to 100 individual images
for i in range(min(args.num_images, 100)):
    img_path = os.path.join(args.output_dir, f'synth_{i:03d}.png')
    # save_image can also write a single [3,H,W] tensor
    save_image(synth[i], img_path, normalize=True, scale_each=True)
print(f"Saved {min(args.num_images,100)} individual images in {args.output_dir}.")

    