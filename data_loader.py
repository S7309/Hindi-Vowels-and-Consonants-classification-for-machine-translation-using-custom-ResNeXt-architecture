#importing required libraries
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

print(torch.cuda.get_device_name(0))

class TamilVowelConsonantDataset(Dataset):
    
    def __init__(self, data_path, transform = None, train = True):
        self.train_img_path = data_path['train']
        self.test_img_path = data_path['test']
        self.train_img_files = os.listdir(self.train_img_path)
        self.test_img_files = os.listdir(self.test_img_path)
        self.transform = transform
        self.train = train
    
    def __len__(self):
        return len(self.train_img_files)
    
    def __getitem__(self, indx):
            
        if self.train:  
            
            if indx >= len(self.train_img_files):
                raise Exception("Index should be less than {}".format(len(self.train_img_files)))
               
            image = Image.open(self.train_img_path + self.train_img_files[indx]).convert('RGB')
            labels = self.train_img_files[indx].split('_')
            V = int(labels[0][1])
            C = int(labels[1][1])
            label = {'Vowel' : V, 'Consonant' : C}

            if self.transform:
                image = self.transform(image)

            return image, label
        
        if self.train == False:
            image = Image.open(self.test_img_path + self.test_img_files[indx]).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image, self.test_img_files[indx]
