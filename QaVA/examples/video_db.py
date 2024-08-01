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

'''
VideoDataset allows you to access frames of a given video.
'''
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_fp, list_of_idxs=[], transform_fn=lambda x: x):
        self.video_fp = video_fp
        self.list_of_idxs = []
        self.transform_fn = transform_fn
        self.cap = swag.VideoCapture(self.video_fp)
        self.video_metadata = json.load(open(self.video_fp + '.json', 'r'))
        self.cum_frames = np.array(self.video_metadata['cum_frames'])
        self.cum_frames = np.insert(self.cum_frames, 0, 0)
        self.length = self.cum_frames[-1]
        self.current_idx = 0
        self.init()
        
    def init(self):
        if len(self.list_of_idxs) == 0:
            self.frames = None
        else:
            self.frames = []
            for idx in tqdm(self.list_of_idxs, desc="Video"):
                self.seek(idx)
                frame = self.read()
                self.frames.append(frame)
            
    def transform(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.transform_fn(frame)
        return frame

    def seek(self, idx):
        if self.current_idx != idx:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
            self.current_idx = idx
        
    def read(self):
        _, frame = self.cap.read()
        frame = self.transform(frame)
        self.current_idx += 1
        return frame
    
    def __len__(self):
        return self.length if len(self.list_of_idxs) == 0 else len(self.list_of_idxs)
    
    def __getitem__(self, idx):
        if len(self.list_of_idxs) == 0:
            self.seek(idx)
            frame = self.read()
        else:
            frame = self.frames[idx]
        return frame   

'''
LabelDataset loads the target dnn .csv files and allows you to access the target dnn outputs of given frames.
'''
class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, labels_fp, length, query_objs, valid_area=None):
        df = pd.read_csv(labels_fp)
        df = df[df['object_name'].isin(query_objs)]
        frame_to_rows = defaultdict(list)
        for row in df.itertuples():
            if valid_area == None or self.is_valid(row, valid_area):
                frame_to_rows[row.frame].append(row)
        labels = []
        for frame_idx in range(length):
            labels.append(frame_to_rows[frame_idx])
        self.labels = labels

    def is_valid(self, row, valid_area):
        xmin, ymin, xmax, ymax = valid_area
        return row.xmin > xmin and row.ymin > ymin and \
            row.xmax < xmax and row.ymax < ymax
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.labels[idx]

'''
Defines our notion of 'closeness' as described in the paper for two labels for only one object type.
'''
def is_close_helper(label1, label2):
    if len(label1) != len(label2):
        return False
    counter = 0
    for obj1 in label1:
        xavg1 = (obj1.xmin + obj1.xmax) / 2.0
        yavg1 = (obj1.ymin + obj1.ymax) / 2.0
        coord1 = [xavg1, yavg1]
        expected_counter = counter + 1
        for obj2 in label2:
            xavg2 = (obj2.xmin + obj2.xmax) / 2.0
            yavg2 = (obj2.ymin + obj2.ymax) / 2.0
            coord2 = [xavg2, yavg2]
            if distance.euclidean(coord1, coord2) < 100:
                counter += 1
                break
        if expected_counter != counter:
            break
    return len(label1) == counter

class AggregateQuery(QaVA.AggregateQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)
    
class LimitQuery(QaVA.LimitQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)
    
class SUPGPrecisionQuery(QaVA.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 0 else 0.0
    
class SUPGRecallQuery(QaVA.SUPGRecallQuery):
    def score(self, target_dnn_output):
        return 1.0 if len(target_dnn_output) > 0 else 0.0
    
class LHSPrecisionQuery(QaVA.SUPGPrecisionQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            mid = 1750 / 2
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                if x < mid:
                    return True
            return False
        return proc_boxes(target_dnn_output)
    
class LHSRecallQuery(QaVA.SUPGRecallQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            mid = 1750 / 2
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                if x < mid:
                    return True
            return False
        return proc_boxes(target_dnn_output)
    
class AveragePositionAggregateQuery(QaVA.AggregateQuery):
    def score(self, target_dnn_output):
        def proc_boxes(boxes):
            avg = 0.
            if len(boxes) == 0:
                return 0.
            for box in boxes:
                x = (box.xmin + box.xmax) / 2.
                avg += x / 1750
            return avg / len(boxes)
        return proc_boxes(target_dnn_output)
    
class ExSampleQuery(QaVA.ExSampleQuery):
    def score(self, target_dnn_output):
        return len(target_dnn_output)
    
    def is_same_obj(self, frame_id1, det1, frame_id2, det2):
        return frame_id1 != frame_id2 and det1.track_id == det2.track_id