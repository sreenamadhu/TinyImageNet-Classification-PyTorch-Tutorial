from torchvision import transforms
from dataset import TinyImageNetDataset
from torch.utils.data import DataLoader
import time


def worker_analysis(ntrain_dataset, num_workers):
    
    train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=num_workers)
    start_time = time.time()
    for i, (images, labels) in enumerate(train_dataloader):
        pass
    end_time = time.time()
    print(f'Time taken for {num_workers} workers: {end_time-start_time:.2f}')
    return

if __name__ == "__main__":
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=64),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TinyImageNetDataset(root_dir = 'tiny-imagenet-200', split = 'train', transform = train_transform)
    for num_workers in [0, 2, 4, 8, 16, 32]:
        worker_analysis(train_dataset, num_workers)