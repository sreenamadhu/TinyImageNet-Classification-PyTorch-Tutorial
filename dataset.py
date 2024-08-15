from torch.utils.data import Dataset
from PIL import Image
import os

class TinyImageNetDataset(Dataset):
    
    def __init__(self, root_dir, split, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.classes = {x.strip('\r\n'): i for i,x in enumerate(open(os.path.join(root_dir, 'wnids.txt'),'r').readlines())}
        root_dir = os.path.join(root_dir, split)
        if split == 'train':
            for folder in os.listdir(root_dir):
                if not os.path.isdir(os.path.join(root_dir, folder)):
                    continue
                for image in os.listdir(os.path.join(root_dir, folder, 'images')):
                    self.image_paths.append(os.path.join(root_dir, folder, 'images', image))
                    self.labels.append(self.classes[folder])
        elif split == 'val':
            label_map = {x.strip('\r\n').split('\t')[0]:self.classes[x.strip('\r\n').split('\t')[1]] for x in open(os.path.join(root_dir, 'val_annotations.txt')).readlines()}
            
            for image in os.listdir(os.path.join(root_dir, 'images')):
                self.image_paths.append(os.path.join(root_dir, 'images', image))  
                self.labels.append(label_map[image])
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        im = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform is not None:
            im = self.transform(im)
        return (im, label)