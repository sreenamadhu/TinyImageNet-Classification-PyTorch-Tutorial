from dataset import TinyImageNetDataset
from train_with_val import train_model
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = TinyImageNetDataset(root_dir = 'tiny-imagenet-200', split = 'train', transform = train_transform)
val_dataset = TinyImageNetDataset(root_dir = 'tiny-imagenet-200', split = 'val', transform = val_transform)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = models.resnet18(pretrained=True)
# Modify the final layer to match Tiny ImageNet classes (200 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 200)

train_model(model, train_dataloader, val_dataloader, num_epochs=100)