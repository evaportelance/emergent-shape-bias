import numpy as np
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#model_dir = Path("./my_amdim/")
#sys.path.append(str(model_dir))
from my_amdim import Checkpointer


class Perception(nn.Module):
    def __init__(self, pretrained_model_dir, pretrained_model_cpt, model_type='amdim'):

        super(Perception, self).__init__()
        self.model_type = model_type

        cpt_load_path = pretrained_model_dir / pretrained_model_cpt
        checkpointer = Checkpointer(str(pretrained_model_dir), pretrained_model_cpt)
        model = checkpointer.restore_model_from_checkpoint(str(cpt_load_path))
        for param in model.parameters():
            param.requires_grad = False
        self.encoder = model

        if model_type == 'amdim':
            self.encoding_size = model.hyperparams['n_rkhs']
        else:
            self.encoding_size = 512


    def forward(self, x):
        if self.model_type == 'amdim':
            x =self.encoder(x1=x, x2=x, encoding_layer=True)
            x = x.view(x.size(0), -1) # b*encoding_size
        else:
            x =self.encoder(x, encoder=True)
            #x = self.encoder(x)
        return x


class Representation(nn.Module):
    def __init__(self, encoding_size, compression_size, hidden_size):

        super(Representation, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(encoding_size, compression_size), nn.Tanh(),
            nn.Linear(compression_size, hidden_size))

    def forward(self, x):
        x = self.encoder(x)
        return x


class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p = 0.7)
        self.act = nn.ReLU()
        self.lin2 = nn.Linear(hidden_size, output_size)
    def forward(self, inputs):
        return self.lin2(self.act(self.norm(self.dropout(self.lin1(inputs)))))


class SimpleCNN(nn.Module):
    def __init__(self, compression_size, hidden_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.norm = nn.BatchNorm2d(50)
        self.fc = TwoLayerMLP(29*29*50, compression_size, hidden_size)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.norm(x)
        x = x.view(x.size(0),-1)
        x =  self.fc(x)
        return x


class Vision(nn.Module):
    def __init__(self, encoding_size = 1024, hidden_size = 12, compression_size = 64, is_cnn = False):
        super(Vision, self).__init__()
        self.is_cnn = is_cnn
        if not is_cnn:
            self.representation = Representation(encoding_size, compression_size, hidden_size)
        else :
            self.representation = SimpleCNN(compression_size, hidden_size)
    def forward(self, x):
        x = self.representation(x) # b*hidden_size
        return x
