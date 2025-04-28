# This is the pretrain or baseline file for cifar10
from __future__ import print_function
import argparse
import os
import math
import gc
import sys
import xlwt
import random
import numpy as np
import foolbox as fb
from foolbox.criteria import Misclassification, TargetedMisclassification
import joblib
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.nn.functional import mse_loss
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torch.utils.data.sampler as sp
from vgg import VGG
from resnet import ResNet18, ResNet50

cudnn.benchmark = True
nz = 128
target = False
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 1000
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--batchSize', type=int, default=128)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save_path', type=str, default='pretrained/resnet50_cifar10.pth')
parser.add_argument('--mode', type=str, default='pretrain', help='Mode: pretrain or dast')
opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: CUDA is available but not being used. Use --cuda to enable it.")

device = torch.device("cuda:0" if opt.cuda and torch.cuda.is_available() else "cpu")

# ------------------ Data ------------------
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# ------------------ Pretrain Mode ------------------
if opt.mode == 'pretrain':
    net = ResNet50(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # ------------------ Training ------------------
    for epoch in range(opt.epochs):
        net.train()
        running_loss = 0.0
        total, correct = 0, 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{opt.epochs} | Loss: {running_loss/len(trainloader):.4f} | Accuracy: {100.*correct/total:.2f}%")

    # ------------------ Save Model ------------------
    torch.save(net.state_dict(), opt.save_path)
    print(f"ResNet-50 trained on CIFAR-10 saved to {opt.save_path}")
