import torch
from torch.utils.data import Dataset


class MessageDataset(Dataset):
    def __init__(self, messages, labels, classifier="shape"):
        self.messages = messages
        self.labels = labels
        self.classifier = classifier

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        message = self.messages[idx]
        label = self.labels[idx]
        sample = {"message": message, "label": label}
        return sample
