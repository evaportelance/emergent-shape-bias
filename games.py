import torch
from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
from enum import IntEnum

class GameType(IntEnum):
    random = 0
    color = 1
    shape = 2
    material = 3
    size = 4


class ClevrGame(Dataset):
    def __init__(self, data_dir, perspective=False, proportion = 1.0, color_balance = False, eval=False):
        super().__init__()

        self.data_dir = data_dir
        self.perspective = perspective
        self.proportion = proportion
        self.color_balance = color_balance
        self.eval = eval
        self.games = []

        if self.eval:
            image_file = data_dir / "test_imgs.npy"
            label_file = data_dir / "test_labels.npy"
        else:
            image_file = data_dir / "train_imgs.npy"
            label_file = data_dir / "train_labels.npy"

        print("\n----------------------------------------")
        print("loading data from {:s}".format(str(image_file)))
        print("----------------------------------------\n")

        with open(image_file, "rb") as f:
            imgs = np.load(f)

        with open(label_file, "rb") as f:
            labels = np.load(f)

        self.imgs = imgs
        self.labels = labels
        self.label_imgs = np.concatenate((labels, imgs), axis=1)


    def create_games(self, num_games, num_distractors, type="random"):
        self.games = []
        allgametype = GameType[type]
        if self.color_balance:
            alternative = GameType["color"]
        else:
            alternative = GameType["random"]
        for i in range(num_games):
            if self.proportion < 1.0 :
                gametype =  allgametype if np.random.random() <= self.proportion else alternative
            else:
                gametype = allgametype
            if gametype == 0: # random
                img_ids = np.random.choice(self.label_imgs.shape[0], num_distractors + 1)
                target_choice_id = np.random.choice(num_distractors + 1)
                target_id = img_ids[target_choice_id]
                target_feature_label = self.label_imgs[target_id][0:4]
                game_imgs = torch.stack([torch.tensor(self.label_imgs[id_][4:]) for id_ in img_ids])
                color_mask = self.label_imgs[:, 0] == target_feature_label[0]
                shape_mask = self.label_imgs[:, 1] == target_feature_label[1]
                material_mask = self.label_imgs[:, 2] == target_feature_label[2]
                size_mask = self.label_imgs[:, 3] == target_feature_label[3]
            else:
                target_choice_id = np.random.choice(num_distractors + 1)
                target_label_id = np.random.choice(self.label_imgs.shape[0])
                target_image = self.label_imgs[target_label_id][4:]
                target_feature_label = self.label_imgs[target_label_id][0:4]
                color_mask = self.label_imgs[:, 0] == target_feature_label[0]
                shape_mask = self.label_imgs[:, 1] == target_feature_label[1]
                material_mask = self.label_imgs[:, 2] == target_feature_label[2]
                size_mask = self.label_imgs[:, 3] == target_feature_label[3]

                if gametype == 1: #color
                    color_game_mask = np.logical_and(np.logical_and(np.logical_and(shape_mask, material_mask), size_mask), np.logical_not(color_mask))
                    imgs_subset = self.label_imgs[color_game_mask]
                elif gametype == 2: #shape
                    shape_game_mask = np.logical_and(np.logical_and(np.logical_and(color_mask, material_mask), size_mask), np.logical_not(shape_mask))
                    imgs_subset = self.label_imgs[shape_game_mask]
                elif gametype == 3: #material
                    material_game_mask = np.logical_and(np.logical_and(np.logical_and(color_mask, shape_mask), size_mask), np.logical_not(material_mask))
                    imgs_subset = self.label_imgs[material_game_mask]
                elif gametype == 4: #size
                    size_game_mask = np.logical_and(np.logical_and(np.logical_and(color_mask, material_mask), shape_mask), np.logical_not(size_mask))
                    imgs_subset = self.label_imgs[size_game_mask]
                elif gametype == 5: #position
                    position_game_mask = np.logical_and(np.logical_and(color_mask, shape_mask), np.logical_and(material_mask, size_mask))
                    imgs_subset = self.label_imgs[position_game_mask]
                img_ids = np.random.choice(imgs_subset.shape[0], num_distractors + 1)
                game_imgs = [imgs_subset[id_][4:] for id_ in img_ids]
                game_imgs[target_choice_id] = target_image
                game_imgs = torch.tensor(game_imgs)
            if self.perspective:
                perspective_game_mask = np.logical_and(np.logical_and(color_mask, shape_mask), np.logical_and(material_mask, size_mask))
                imgs_subset = self.label_imgs[perspective_game_mask]
                target_id = np.random.choice(imgs_subset.shape[0])
                target_img = torch.tensor(imgs_subset[target_id][4:])

            else:
                target_img = game_imgs[target_choice_id]
            self.games.append({"imgs": game_imgs.float(),
                          "labels": target_choice_id,
                          "target_img": target_img.float()})
    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        return self.games[idx]
