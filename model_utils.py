import torch
import torch.nn as nn
import torch.nn.functional as F
import edit_distance

def find_lengths_onehot(messages: torch.Tensor, vocab_size) -> torch.Tensor:
    zero = F.one_hot(torch.tensor([0]), vocab_size).float()[0]
    zero = zero.to(messages.device)

    max_k = messages.size(1)
    zero_mask = messages == zero
    zero_mask = zero_mask.sum(dim=-1) == vocab_size

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths

def find_lengths(messages: torch.Tensor) -> torch.Tensor:
    max_k = messages.size(1)
    zero_mask = messages == 0

    lengths = max_k - (zero_mask.cumsum(dim=1) > 0).sum(dim=1)
    lengths.add_(1).clamp_(max=max_k)

    return lengths

def get_edit_distance(ref, hyp):
    sm = edit_distance.SequenceMatcher(a=list(ref), b=list(hyp))
    return sm.distance()

def add_eos_to_messages(message, message_length, max_length):
    cleaned_message = torch.zeros_like(message)

    for i in range(max_length):
        not_eosed = (i < message_length).float()
        cleaned_message[:, i] = message[:, i] * not_eosed
    return cleaned_message

class MessageClassifier(nn.Module):
        def __init__(self, n_classes, message_max_length):

            super(MessageClassifier, self).__init__()

            self.linearlayer =nn.Linear(message_max_length, n_classes, bias=True)

        def forward(self, x):
            x = self.linearlayer(x)
            return x

class MessageMLP(nn.Module):
        def __init__(self, n_classes, message_max_length, hidden_size=10):

            super(MessageMLP, self).__init__()
            self.mlp =nn.Sequential(nn.Linear(message_max_length, hidden_size, bias=False),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_size, n_classes, bias=True))

        def forward(self, x):
            x = self.mlp(x)
            return x
