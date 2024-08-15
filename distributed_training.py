import torch.nn as nn
import torch.optim as optim
import torch
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torchvision.models import resnet18, ResNet18_Weights

def model_setup(rank):
    resnet18 = resnet18(weights=ResNet18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 200)
    device = torch.device(f'cuda:{rank}')
    model.to(device)
    model = DDP(model, device_ids=[rank])
    return model

def distributed_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def create_dataloader(rank, world_size):
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=64),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_dataset = TinyImageNetDataset(root_dir = 'tiny-imagenet-200', split = 'train', transform = train_transform)
    sampler = distributed.DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank = rank)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, sampler=sampler)
    return train_dataloader

def train_model(rank, world_size):
    distributed_setup(rank, world_size)
    model = model_setup(rank)
    train_dataloader = create_dataloader(rank, world_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    logs_f = open('logs_distributed.txt','w+')
    model.train()
    for epoch in range(100):
        running_loss = 0.0
        corrects = 0
        total = 0
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.cuda(rank), labels.cuda(rank)
            # zero the parameter gradients
            optimizer.zero_grad()   
            # Forward pass: Pass image data from training dataset, make predictions about class image belongs to (0-9 in this case)
            outputs = model(inputs)

            # Define our loss function, and compute the loss
            loss = criterion(outputs, labels)
            # Backward pass: compute the gradients of the loss w.r.t. the model's parameters
            loss.backward()
            # Update the neural network weights
            optimizer.step()

            # Log statistics
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
            running_loss += loss.item() * inputs.size(0)
            if i % 200 == 199:    # Log every 200 mini-batches
                logs_f.write(f'Rank #{rank} Epoch #{epoch+1} Iteration #{i + 1:5d} loss: {running_loss / len(train_dataloader.dataset) :.3f}\n')
                print(f'Rank #{rank} Epoch #{epoch+1} Iteration #{i + 1:5d} loss: {running_loss / len(train_dataloader.dataset) :.3f}\n')
        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        epoch_train_acc = corrects/float(total)
    logs_f.close()
    cleanup()
    return

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_model,
             args=(world_size,),
             nprocs=world_size,
             join=True)