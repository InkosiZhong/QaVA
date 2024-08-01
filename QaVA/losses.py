import torch
import torch.nn as nn
import torch.nn.functional as F
from tabulate import tabulate

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()
    
def union_weight_fn(unique_labels, class_probs):
    return torch.ones_like(class_probs) / len(class_probs)

def head_prior_weight_fn(unique_labels, class_probs):
    return class_probs

def tail_prior_weight_fn(unique_labels, class_probs):
    #return 1 - class_probs
    return torch.log(class_probs.max() / class_probs) + 1

def high_score_prior_weight_fn(unique_labels, class_probs):
    weight = torch.sqrt(torch.tensor(unique_labels).cuda() + 1)
    return weight
    
def get_weight_from_labels(labels, unique_labels, weight_fn=union_weight_fn, mask=None):
    labels = torch.from_numpy(labels.astype(float)).to(int)
    class_probs = torch.bincount(labels)[unique_labels] / len(labels)
    weight = weight_fn(unique_labels, class_probs)
    if mask is not None:
        weight *= mask
    weight /= weight.sum()
    return weight.to(torch.float32)

class ReweightedCrossEntropyLoss(nn.Module):
    def __init__(self, labels, unique_labels, weight_fn, mask=None) -> None:
        super().__init__()
        weight = get_weight_from_labels(labels, unique_labels, weight_fn, mask)
        table_data = [
            ['class'] + [x for x in unique_labels],
            ['weight'] + [f'{x:.2f}' for x in weight.tolist()]
        ]
        print('ReweightedCrossEntropyLoss:')
        print(tabulate(table_data, tablefmt='fancy_grid'))

        self.loss_fn = nn.CrossEntropyLoss(weight)
        
    def forward(self, pred, label):
        return self.loss_fn(pred, label)

class FocalLoss(nn.Module):
    def __init__(self, labels, unique_labels, weight_fn, gamma=2, mask=None):
        super().__init__()
        self.register_buffer('alpha', get_weight_from_labels(labels, unique_labels, weight_fn, mask))
        table_data = [
            ['class'] + [x for x in unique_labels],
            ['alpha'] + [f'{x:.2f}' for x in self.alpha.tolist()]
        ]
        print('ReweightedCrossEntropyLoss:')
        print(tabulate(table_data, tablefmt='fancy_grid'))
        self.gamma = gamma

    def forward(self, pred, label):
        label = label.to(int)
        alpha = self.alpha[label]
        log_softmax = torch.log_softmax(pred, dim=1)
        logpt = torch.gather(log_softmax, dim=1, index=label.view(-1, 1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)

class HuberLoss(nn.Module):
    def __init__(self, labels, unique_labels, weight_fn, mask=None):
        super().__init__()
        self.huber = torch.nn.HuberLoss(reduction='sum')
        self.unique_labels = unique_labels
        self.weight = get_weight_from_labels(labels, unique_labels, weight_fn, mask)
        table_data = [
            ['class'] + [x for x in unique_labels],
            ['weight'] + [f'{x:.2f}' for x in self.weight.tolist()]
        ]
        print('HuberLoss:')
        print(tabulate(table_data, tablefmt='fancy_grid'))

    def forward(self, pred, label):
        loss = 0
        for l, w in zip(self.unique_labels, self.weight):
            idx = label == l
            loss += w * self.huber(pred[idx], label[idx])
        return loss / len(label)

if __name__ == '__main__':
    import numpy as np
    x = torch.rand(3)
    y = torch.tensor([0, 0, 1])
    loss = HuberLoss(np.array([0, 0, 1]), np.array([0, 1]))
    print(loss(x, y))