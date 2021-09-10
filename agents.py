import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

import os

from vision import Vision
from model_utils import find_lengths, find_lengths_onehot


class Agent(nn.Module):
    def __init__(self, agent_hps, vision_hps, device):
        super(Agent, self).__init__()
        self.device = device
        # agent hps
        self.hidden_size = agent_hps["hidden_size"]
        self.hidden_mlp = agent_hps["hidden_mlp"]
        self.vocab_size = agent_hps["vocab_size"]
        self.emb_size = agent_hps["emb_size"]
        self.message_length = agent_hps["message_length"]
        # to keep track of individual agent progress
        self.epochs_community = 0
        self.epochs_teacher = 0
        self.epochs_selfplay = 0

        # shared embedding layer
        self.shared_embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.sos_embedding = nn.Parameter(torch.zeros(self.emb_size))

        # vision layers
        self.encoding_size = vision_hps["encoding_size"]
        self.compression_size = vision_hps["compression_size"]
        vision = Vision(self.encoding_size, self.hidden_size, self.compression_size)
        self.vision = vision.to(self.device)

        # sender modules
        self.sender_decoder = nn.LSTMCell(self.emb_size, self.hidden_size)
        self.sender_hidden_to_output = nn.Linear(self.hidden_size, self.vocab_size)

        # receiver modules
        self.receiver_encoder = nn.LSTM(self.emb_size, self.hidden_size, batch_first=True)
        self.receiver_hidden_to_output = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_mlp), nn.ReLU(), nn.Linear(self.hidden_mlp, self.hidden_size))


    def forward(self, mode, **kwargs):
        if mode == "sender":
            output = self.sender_forward(tgt_img=kwargs["tgt_img"])  # message

        elif mode == "receiver":
            output = self.receiver_forward(imgs=kwargs["imgs"], message=kwargs["message"])  # prediction

        elif mode == "selfplay":
            output = self.selfplay_forward(tgt_img=kwargs["tgt_img"], imgs=kwargs["imgs"]) # message and prediction

        return output

    def sender_forward(self, tgt_img):
        ht = self.vision(tgt_img)
        batch_size = ht.size(0)

        ct = torch.zeros_like(ht)
        et = torch.stack([self.sos_embedding] * batch_size)

        message = []
        log_probs = []
        entropy = []
        probs = []

        for i in range(self.message_length - 1):
            ht, ct = self.sender_decoder(et, (ht, ct))
            step_logits = self.sender_hidden_to_output(ht)
            #step_probs = F.gumbel_softmax(step_logits, hard=False, dim=1)
            step_probs = F.softmax(step_logits, -1)
            distr = Categorical(probs=step_probs)

            if self.training:
                token = distr.sample()
            else:
                token = step_logits.argmax(dim=1)


            et = self.shared_embedding(token)

            message.append(token)
            log_probs.append(distr.log_prob(token))
            entropy.append(distr.entropy())
            probs.append(step_probs)

        ## add eos probs to all in batch
        # eos_probs = torch.zeros_like(probs[0])
        # idx = torch.zeros((batch_size), dtype=torch.long)
        # idx = idx.to(eos_probs.device)
        # eos_probs[torch.arange(batch_size),idx] = 1
        # probs.append(eos_probs)

        message = torch.stack(message).permute(1, 0)
        log_probs = torch.stack(log_probs).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)
        all_probs = torch.stack(probs).permute(1,0,2)

        ## adds a zero to the end of each message in the batch. zero the eos character
        zeros = torch.zeros((message.size(0), 1)).to(message.device)
        message = torch.cat([message, zeros.long()], dim=1)
        log_probs = torch.cat([log_probs, zeros], dim=1)
        entropy = torch.cat([entropy, zeros], dim=1)

        return message, log_probs, entropy, all_probs

    def receiver_forward(self, message, imgs):
        batch_size = message.size(0)
        num_imgs = imgs.size(1)
        ### treat this batch of sets as one big batch collapsing in set size dimension
        imgs = imgs.view(batch_size*num_imgs, self.encoding_size)

        feature_vectors = self.vision(imgs)
        feature_vectors = feature_vectors.view(batch_size, num_imgs, -1) # b * num_imgs * 1 * self.hidden_size

        ### takes message and gets representations for each token and returns final hidden state which is the models understanding
        emb = self.shared_embedding(message)
        lengths = find_lengths(message)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)

        states, rnn_hidden = self.receiver_encoder(packed)

        #states_all, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(states, batch_first=True)
        #all_outputs = self.receiver_hidden_to_output(all_states)
        #all_logits = torch.einsum('bnd,bmd->bmn', feature_vectors, states_all)
        #all_probs = F.softmax(all_logits, -1) # p(image_i | message_<=j)
        all_logits = None

        emb_msg = rnn_hidden[1].view(batch_size, self.hidden_size) #(B,D)
        emb_msg = self.receiver_hidden_to_output(emb_msg)
        logits = torch.einsum('bnd,bd->bn', feature_vectors, emb_msg)
        probs = F.softmax(logits, -1)  # B * N,  p(image_i | message)

        distr = Categorical(probs=probs)

        if self.training:
            choice = distr.sample()

        else:
            choice = probs.argmax(dim=1)

        log_probs_choices = distr.log_prob(choice)
        entropy = None

        return choice, log_probs_choices, entropy, logits, all_logits

    def selfplay_forward(self, tgt_img, imgs):
        #Sender
        ht = self.vision(tgt_img)

        ct = torch.zeros_like(ht)
        et = torch.stack([self.sos_embedding] * ht.size(0))

        message = []
        log_probs = []
        entropy = []

        for i in range(self.message_length - 1):
            ht, ct = self.sender_decoder(et, (ht, ct))

            step_logits = self.sender_hidden_to_output(ht)
            # Use gumbel softmax trick to get discreet tokens for message that are still differentiable
            token = F.gumbel_softmax(step_logits, hard=True, dim=1)
            step_probs = F.softmax(step_logits, -1)
            distr = Categorical(probs=step_probs)

            et = torch.matmul(token, self.shared_embedding.weight)

            message.append(token)
            log_probs.append(distr.log_prob(token.argmax(dim=1)))
            entropy.append(distr.entropy())

        message = torch.stack(message).permute(1, 0, 2)
        log_probs = torch.stack(log_probs).permute(1, 0)
        entropy = torch.stack(entropy).permute(1, 0)

        # add filler for eos char
        zeros = torch.zeros((message.size(0), 1)).to(message.device)
        s_log_probs = torch.cat([log_probs, zeros], dim=1)
        s_entropy = torch.cat([entropy, zeros], dim=1)

        zeros = torch.zeros(message.size(0), dtype = torch.long)
        eos_char = F.one_hot(zeros, self.vocab_size).float().unsqueeze(1).to(message.device)
        message = torch.cat([message, eos_char], dim=1)

        # Receiver
        batch_size = message.size(0)
        num_imgs = imgs.size(1)
        ### treat this batch of sets as one big batch collapsing in set size dimension
        imgs = imgs.view(batch_size*num_imgs, self.encoding_size)

        feature_vectors = self.vision(imgs)
        feature_vectors = feature_vectors.view(batch_size, num_imgs, -1) # b * num_imgs * 1 * self.hidden_size

        ### takes message and gets representations for each token and returns final hidden state which is the models understanding
        emb = torch.matmul(message, self.shared_embedding.weight)
        lengths = find_lengths_onehot(message, self.vocab_size)

        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, rnn_hidden = self.receiver_encoder(packed)

        emb_msg = rnn_hidden[1].view(batch_size, self.hidden_size) #(B,D)
        # emb_msg = self.receiver_hidden_to_output(emb_msg)
        logits = torch.einsum('bnd,bd->bn', feature_vectors, emb_msg)
        choices_probs = F.softmax(logits, -1)
        max_choice = choices_probs.argmax(dim=1)

        #transform message from vector of one hot vectors to vector of indices for message similarity
        message_onehot = message.detach().cpu()
        message = np.argmax(message_onehot, axis=2)

        return message, logits, max_choice, s_log_probs, s_entropy


### For MI eval

    def get_message_firstembedding(self, tgt_img):
        ht = self.vision(tgt_img)
        batch_size = ht.size(0)

        ct = torch.zeros_like(ht)
        et = torch.stack([self.sos_embedding] * batch_size)

        message = []
        first_embedding = None

        for i in range(self.message_length - 1):
            ht, ct = self.sender_decoder(et, (ht, ct))
            step_logits = self.sender_hidden_to_output(ht)
            #step_probs = F.gumbel_softmax(step_logits, hard=False, dim=1)
            step_probs = F.softmax(step_logits, -1)
            distr = Categorical(probs=step_probs)

            token = step_logits.argmax(dim=1)
            et = self.shared_embedding(token)
            if i == 0:
                first_embedding = et

            message.append(token)

        message = torch.stack(message).permute(1, 0)

        ## adds a zero to the end of each message in the batch. zero the eos character
        zeros = torch.zeros((message.size(0), 1)).to(message.device)
        message = torch.cat([message, zeros.long()], dim=1)

        return message, first_embedding
