import os
from enum import Enum

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

INTERP = 3


class TransformsImage128:
    '''
    Image dataset, for use with 128x128 full image encoder.
    '''
    def __init__(self):
        # image augmentation functions
        self.flip_lr = transforms.Compose([transforms.ToPILImage(),
                                           transforms.RandomHorizontalFlip(p=0.5)])
        rand_crop = \
            transforms.RandomResizedCrop(128, scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=INTERP)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        post_transform = transforms.Compose([
            transforms.ToTensor(),
            # Removing normalization for now as it seems to remove a lot and sometimes make images all black, will try to train without.
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        self.test_transform = transforms.Compose([
            transforms.ToPILImage(),
            #transforms.Resize(146, interpolation=INTERP),
            #transforms.CenterCrop(128),
            post_transform
        ])
        self.train_transform = transforms.Compose([
            rand_crop,
            col_jitter,
            rnd_gray,
            post_transform
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2
    
class MujocoDataset(Dataset):
    
    def __init__(self,  input_dir, transform, train=True, dataset_whole=True, save_dir=None):
        super().__init__()

        self.data_dir = input_dir
        self.transform = transform
        self.train = train
        self.dataset_whole = dataset_whole
        self.save_dir = save_dir
        self.idx_to_imgs = {}
        self.idx_to_transformed_items = {}
        

        self.create()
        
        
    def get_idx(self, imgs, labels):   

        def get_category_index(label):
            lab = np.asarray(label).nonzero()
            colour = lab[0][0]
            shape = lab[0][1] - 8
            category = colour + (8 * shape)
            return (category, colour, shape)

        for id_ in range(0, labels.shape[0]):
            img = imgs[id_]
            cat_id = get_category_index(labels[id_]) 
            self.idx_to_imgs[id_] = img, cat_id
            self.idx_to_transformed_items[id_] = self.transform(img), cat_id

            
    def create(self):
        if self.train:
            with open(os.path.join(self.data_dir,"images_train.npy"), "rb") as f:
                imgs = np.load(f)
            with open(os.path.join(self.data_dir,"rewards_train.npy"), "rb") as f:
                labels = np.load(f)
        else:
            with open(os.path.join(self.data_dir,"images_test.npy"), "rb") as f:
                imgs = np.load(f)
            with open(os.path.join(self.data_dir,"rewards_test.npy"), "rb") as f:
                labels = np.load(f)
            
        
        if self.dataset_whole:
            self.get_idx(imgs, labels)
        else:
            self.get_idx(imgs[0:500], labels[0:500])
            
    
    def __len__(self):
        return len(self.idx_to_transformed_items)

    
    def __getitem__(self, idx):
        return self.idx_to_transformed_items[idx]
    

def build_dataset(batch_size, input_dir): 

    train_transform = TransformsImage128()
    test_transform = train_transform.test_transform
    train_dataset = MujocoDataset(input_dir, train_transform)
    test_dataset = MujocoDataset(input_dir, test_transform, train=False)

    # build pytorch dataloaders for the datasets
    train_loader = \
        torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True
                                    #num_workers=16
                                   )
    test_loader = \
        torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    drop_last=True
                                    #num_workers=16
                                   )

    return train_loader, test_loader

