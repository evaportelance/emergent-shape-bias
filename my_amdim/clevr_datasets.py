import os
import glob
import json

from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

INTERP = 3

class TransformsClevr:
    '''
    Image dataset, for use with 64x96 clevr image encoder.
    '''
    def __init__(self):
        # image augmentation functions
        self.flip_lr = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
        rand_crop = \
            transforms.RandomResizedCrop((64,96), scale=(0.3, 1.0), ratio=(0.7, 1.4),
                                         interpolation=INTERP)
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        post_transform = transforms.Compose([
            transforms.Resize((128,128), interpolation=INTERP),
            transforms.ToTensor(),
            # Removing normalization for now as it seems to remove a lot and sometimes make images all black, will try to train without.
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
        self.test_transform = transforms.Compose([
            #transforms.Resize(128, interpolation=INTERP),
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


class ClevrDataset(Dataset):

    def __init__(self,  input_dir, transform, train=True, save_dir=None):
        super().__init__()

        self.data_dir = input_dir
        self.transform = transform
        self.train = train
        self.save_dir = save_dir
        self.idx_to_imgs = {}
        self.idx_to_transformed_items = {}


        self.create()


    def get_idx(self, imgs, labels):

        def get_category_index(label):
            with open(os.path.join(self.data_dir,"properties-dict.json")) as f:
                label_idx_dict = json.load(f)
            features = []
            for object in label["objects"]:
                colour = label_idx_dict['colors'][object["color"]]
                shape = label_idx_dict['shapes'][object["shape"]]
                material = label_idx_dict['materials'][object["material"]]
                size = label_idx_dict['sizes'][object["size"]]
                features.append((colour, shape, material, size))
            # For now just return features of first shape
            return features[0]

        for id_, (img, label) in enumerate(zip(imgs, labels)):
            cat_id = get_category_index(labels[id_])
            self.idx_to_imgs[id_] = img.convert('RGB'), cat_id
            self.idx_to_transformed_items[id_] = self.transform(img.convert('RGB')), cat_id


    def create(self):

        def label_loader(file):
            with open(file) as json_file:
                data = json.load(json_file)
            return data

        imgs = []
        labels = []

        if self.train:
            for img in glob.glob(self.data_dir+'images/train/*.png'):
                id = img[img.rfind('/'):-4]
                imgs.append(Image.open(img).convert('RGB'))
                labels.append(label_loader(self.data_dir+'scenes/train/{}.json'.format(id)))

        else:
            # for img in glob.glob(self.data_dir+'images/test/*.png'):
            #     id = img[img.rfind('/'):-4]
            #     imgs.append(Image.open(img).convert('RGB'))
            #     labels.append(label_loader(self.data_dir+'scenes/test/{}.json'.format(id)))
            for img in glob.glob(self.data_dir+'images/train/*.png'):
                id = img[img.rfind('/'):-4]
                imgs.append(Image.open(img).convert('RGB'))
                labels.append(label_loader(self.data_dir+'scenes/train/{}.json'.format(id)))

        self.get_idx(imgs, labels)



    def __len__(self):
        return len(self.idx_to_transformed_items)


    def __getitem__(self, idx):
        return self.idx_to_transformed_items[idx]


def build_dataset(batch_size, input_dir):

    train_transform = TransformsClevr()
    test_transform = train_transform.test_transform
    train_dataset = ClevrDataset(input_dir, train_transform)
    test_dataset = ClevrDataset(input_dir, test_transform, train=False)

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
