import os
from pathlib import Path
import click
import torch
import numpy as np
from torchvision import transforms
import json
import glob
from PIL import Image

from vision import Perception
from my_amdim import Checkpointer

INTERP = 3

@click.command()
@click.option("--data-dir", default="../game_data/clevr_shape_split/")
@click.option("--pretrained-model-dir", default="./my_amdim/runs/")
@click.option("--pretrained-model-cpt", default="amdim_clevr_shape_all_200b_500e.pth")
@click.option("--model-type", default="amdim")
@click.option("--device", default="cuda")
@click.option("--test", is_flag=True)
@click.option("--img-size", default=128)
@click.option("--batch-size", default=200)

def run(data_dir, pretrained_model_dir, pretrained_model_cpt, model_type, device, test, img_size, batch_size):
    data_dir = Path(data_dir)
    pretrained_model_dir = Path(pretrained_model_dir)

    with open(data_dir / "properties-dict.json") as f:
        label_idx_dict = json.load(f)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=INTERP),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    perception = Perception(pretrained_model_dir, pretrained_model_cpt, model_type)
    perception = perception.to(device)

    imgs = []
    labels = []

    if test:
        image_path = data_dir / Path('images/test/')
        scene_path = data_dir / Path('scenes/test/')
        img_file = "test_imgs.npy"
        label_file = "test_labels.npy"
    else:
        image_path = data_dir / Path('images/train/')
        scene_path = data_dir / Path('scenes/train/')
        img_file = "train_imgs.npy"
        label_file = "train_labels.npy"

    imgs_batch = []
    for image in glob.glob(str(image_path.joinpath('*.png'))):
        # get image and transform
        id = image[-20:-4]
        with Image.open(image) as img:
            img_transform = transform(img.convert('RGB'))
            imgs_batch.append(img_transform)
        # get corresponding label
        json_file = '{}.json'.format(id)
        labels.append(label_loader(scene_path.joinpath(json_file), label_idx_dict))

        if len(imgs_batch) == batch_size:
            imgs_stack = torch.stack(imgs_batch)
            imgs_stack = imgs_stack.to(device)
            with torch.no_grad():
                feats = perception(imgs_stack)
            imgs.append(feats)
            imgs_batch = []
    if len(imgs_batch) > 0:
        imgs_stack = torch.stack(imgs_batch)
        imgs_stack = imgs_stack.to(device)
        with torch.no_grad():
            feats = perception(imgs_stack)
        imgs.append(feats)

    imgs = torch.cat(imgs)
    imgs = imgs.cpu().numpy()

    labels = np.array(labels)

    np.save(data_dir.joinpath(img_file), imgs)
    np.save(data_dir.joinpath(label_file), labels)

def label_loader(file, label_idx_dict):
    with open(file) as json_file:
        data = json.load(json_file)
        features = []
    for object in data["objects"]:
        colour = label_idx_dict['colors'][object["color"]]
        shape = label_idx_dict['shapes'][object["shape"]]
        material = label_idx_dict['materials'][object["material"]]
        size = label_idx_dict['sizes'][object["size"]]
        features.append([colour,shape,material,size])
    cat = features[0]
    return cat

if __name__ == "__main__":
    run()
