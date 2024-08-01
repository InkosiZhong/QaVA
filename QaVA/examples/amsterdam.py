'''
This code allows you to reproduce the results in the paper corresponding to the "night-street" dataset.
The term 'offline' refers to the fact that all the target dnn outputs have already been computed.
If you like to run the 'online' version (target dnn runs in realtime), take a look at "night_street_online.py". 
Look at the README.md file for information about how to get the data to run this code.
'''
import os
import cv2
import swag
import json
import QaVA
import torch
import pandas as pd
import numpy as np
import torchvision
from scipy.spatial import distance
from collections import defaultdict
from tqdm.autonotebook import tqdm
from QaVA.examples.video_db import VideoDataset, LabelDataset
from QaVA.examples.video_db import AggregateQuery
from QaVA.examples.video_db import LimitQuery
from QaVA.examples.video_db import SUPGPrecisionQuery
from QaVA.examples.video_db import SUPGRecallQuery
from QaVA.examples.video_db import LHSPrecisionQuery
from QaVA.examples.video_db import LHSRecallQuery
from QaVA.examples.video_db import AveragePositionAggregateQuery
from QaVA.examples.video_db import ExSampleQuery
from QaVA.model import *
from QaVA.losses import union_weight_fn, head_prior_weight_fn, tail_prior_weight_fn

# Feel free to change this!
ROOT_DATA_DIR = '/home/inkosizhong/Lab/VideoQuery/datasets/amsterdam'

'''
Preprocessing function of a frame before it is passed to the Embedding DNN.
'''
def embedding_dnn_transform_fn(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame

def target_dnn_transform_fn(frame):
    frame = torchvision.transforms.functional.to_tensor(frame)
    return frame
        
class AmsterdamOfflineIndex(QaVA.Index):
    def get_target_dnn(self):
        '''
        In this case, because we are running the target dnn offline, so we just return the identity.
        '''
        model = torch.nn.Identity()
        return model
        
    def get_score_dnn(self):
        if self.config.index_model == None:
            embedding_size = 512
        else:
            embedding_size = 128

        self.config.score_dnn_kwargs.update(dict(
            encoder_config=self.config.encoder_config, 
            mlp_config=self.config.mlp_config,
            act_layer=self.config.act_layer,
            embedding_size=embedding_size,
            unique_labels=self.unique_labels
        ))

        dnn_type = self.config.score_dnn
        model = dnn_type(**self.config.score_dnn_kwargs)
        
        return model
    
    def get_pretrained_embedding_dnn(self):
        '''
        Note that the pretrained embedding dnn sometime differs from the embedding dnn.
        '''
        model = torchvision.models.resnet18(pretrained=True, progress=True)
        if self.config.index_model == None:
            model.fc = torch.nn.Identity()
        else:
            model.fc = torch.nn.Linear(512, 128)
            model.load_state_dict(torch.load(self.config.index_model))
        return model
    
    def get_target_dnn_dataset(self, train_or_test):
        if train_or_test == 'train':
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-04-10')
        else:
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-04-11')
        video = VideoDataset(
            video_fp=video_fp,
            transform_fn=target_dnn_transform_fn
        )
        return video
    
    def get_embedding_dnn_dataset(self, train_or_test):
        if train_or_test == 'train':
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-04-10')
        else:
            video_fp = os.path.join(ROOT_DATA_DIR, '2017-04-11')
        video = VideoDataset(
            video_fp=video_fp,
            transform_fn=embedding_dnn_transform_fn
        )
        return video
    
    def override_target_dnn_cache(self, target_dnn_cache, train_or_test):
        if train_or_test == 'train':
            labels_fp = '/home/inkosizhong/Lab/VideoQuery/datasets/blazeit/filtered/amsterdam/amsterdam-2017-04-10.csv'
        else:
            labels_fp = '/home/inkosizhong/Lab/VideoQuery/datasets/blazeit/filtered/amsterdam/amsterdam-2017-04-11.csv'
        labels = LabelDataset(
            labels_fp=labels_fp,
            length=len(target_dnn_cache),
            query_objs=self.config.query_objs
        )
        return labels
    
    def score_fn(self, target_dnn_output):
        return len(target_dnn_output)
    
class AmsterdamOfflineConfig(QaVA.IndexConfig):
    def __init__(self):
        super().__init__()
        self.cache_root = 'cache'
        self.index_model = 'cache/tasti.pt'
        self.query_objs = ['bicycle']

        self.do_offline_indexing = True
        self.do_mining = True
        self.do_propagation = True
        self.do_training = True
        self.do_infer = True
        
        self.batch_size = 1024
        self.nb_train = 3000
        self.train_lr = 1e-3
        self.max_training_epochs = 500
        self.early_stop = 2
        self.val_rate = 0.1
        self.max_k = 5

        self.score_dnn = ScoreNet
        self.score_dnn_kwargs = {}
        self.encoder_config = {
            'otype': 'DenseGrid',
            'n_levels': 16,
            'n_features_per_level': 4,
            'log2_hashmap_size': 32,
            'base_resolution': 16,
            'per_level_scale': 1.35
        }
        self.mlp_config = {
            'otype': 'FullyFusedMLP', # FullyFusedMLP, CutlassMLP
            'activation': 'ReLU',
            'output_activation': 'None',
            'n_neurons': 512,
            'n_hidden_layers': 5
        }
        self.act_layer = 'None' # ['None', 'Sigmoid', 'Softmax']

def fit(y_pred, y_true):
    cnt = 0
    for p, t in zip(y_pred, y_true):
        if t - 0.5 < p < t + 0.5:
            cnt += 1
    return cnt / len(y_true)
    
if __name__ == '__main__':
    config = AmsterdamOfflineConfig()
    config.query_objs = ['bicycle']
    config.act_layer = 'None'
    config.score_dnn = ScoreNet
    config.weight_fn = head_prior_weight_fn
    index = AmsterdamOfflineIndex(config)
    index.init()
    query = AggregateQuery(index)
    query.execute_metrics(err_tol=0.01, confidence=0.05, seed=0, step=1000)

    config.weight_fn = QaVA.losses.tail_prior_weight_fn
    index = AmsterdamOfflineIndex(config)
    index.init()
    query = LimitQuery(index)
    res = query.execute_metrics(want_to_find=6, nb_to_find=7, step=1000, min_sample=5)