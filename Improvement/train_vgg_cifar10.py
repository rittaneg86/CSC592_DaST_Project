import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg import VGG  # Your custom VGG model with VGG11/13/16/19 support

# Configuration
model_name = 'VGG16'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# CIFAR-10 Datasets and Loaders
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)

# Convert test_loader to a full tensor (xTest, yTest)
def dataloader_to_tensor(dataloader):
    x_list, y_list = [], []
    for x, y in dataloader:
        x_list.append(x)
        y_list.append(y)
    x_tensor = torch.cat(x_list, dim=0)
    y_tensor = torch.cat(y_list, dim=0)
    return x_tensor, y_tensor

xTest, yTest = dataloader_to_tensor(test_loader)

# Print min and max of xTest
print("Max of xTest:", xTest.max().item())
print("Min of xTest:", xTest.min().item())

# Initialize model
model = VGG(model_name).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Best validation tracking
best_val_acc = 0.0

# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
       
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
       
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_acc = 100. * correct / total
    avg_train_loss = total_loss / len(train_loader)

    # Validation/Test evaluation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total
    avg_val_loss = val_loss / len(test_loader)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}% || "
          f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), f'vgg_{model_name.lower()}_best.pth')
        print(f" Best model saved at Epoch {epoch+1} with Val Acc: {val_acc:.2f}%")

# Save final model
torch.save(model.state_dict(), f'vgg_{model_name.lower()}_final.pth')
print(f' Final model saved as vgg_{model_name.lower()}_final.pth')
