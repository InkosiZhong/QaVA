import cv2
import swag
import json
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
from tqdm.autonotebook import tqdm

class ScoreDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            embeddings,
            list_of_idxs,   # reps
            labels,         # regression, for current set (train or val)
            label_idx,      # classify, for current set (train or val), inv
            preds=None,     # propagation result
            is_classify=False
    ):
        self.embeddings = embeddings[list_of_idxs]
        t_coord = torch.linspace(0, 1, len(embeddings))[list_of_idxs]
        if preds is not None:
            self.preds = preds[list_of_idxs]
            pred_coord = torch.from_numpy((self.preds - self.preds.min()) / (self.preds.max() - self.preds.min()))
            #pred_coord = torch.from_numpy(self.preds)
        else:
            self.preds = None
            pred_coord = torch.zeros_like(t_coord)
        self.coord = torch.stack((t_coord, pred_coord), dim=-1)

        if is_classify:
            self.labels = label_idx
        else: # regression
            self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        coord = self.coord[idx]
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        if self.preds is not None:
            pred = self.preds[idx]
        else: # only for debug
            pred = 0
        return coord, embedding, label, pred