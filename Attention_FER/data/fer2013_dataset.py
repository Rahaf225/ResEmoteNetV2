import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class EnhancedFER2013(Dataset):
    def __init__(self, csv_file, img_dir, augment=False):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.augment = augment
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.augment_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_name).convert('L')
        
        if image.size != (48, 48):
            image = image.resize((48, 48))
        
        if self.augment:
            image = self.augment_transform(image)
        else:
            image = self.transform(image)
            
        label = self.labels.iloc[idx, 1]
        return image, label