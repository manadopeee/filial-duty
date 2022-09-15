import os
from typing import Tuple, List, Sequence, Callable, Dict
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import KeypointRCNN

# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# label = 'take_medicine'
csv_path = f'./data/hyodol/threshold/keypoint/P051-P060/P059/A031_P059_G002_C003.csv'
frame_num = 91
print(f'csv_name : {csv_path.split("/")[-1]}, frame_num : {frame_num}')

# img_path = f'D:/DATASET/hyodol/OUTPUT_images/{label}/{file_name}.jpg'
img_path = os.path.join('/'.join(csv_path.split('/')[:4]), 'video', '/'.join(csv_path.split('/')[5:]).split('.')[0])
img_path = f'{img_path}_frame_{frame_num - 1}.jpg'

# print(f'img_path : {img_path}')
# print(f'csv_path : {csv_path}')

df = pd.read_csv(csv_path)
# print(df.head())
# # df_keypoints = df[]
# # df_keypoint_column = list(df)
df_keypoint_column = []
for column in df:
    if '_x' in column or '_y' in column:
        df_keypoint_column.append(column)

df_keypoints = df[df_keypoint_column]
df_keypoints.loc[1].shape
df_keypoints.loc[frame_num]
# print(df_keypoints.loc[frame_num])

df_bbox_column = []
for column in df:
    if 'bbox' in column and 'score' not in column:
        df_bbox_column.append(column)

df_bbox = df[df_bbox_column]
df_bbox.loc[1].shape
df_bbox.loc[frame_num]
# print(df_bbox.loc[frame_num])

def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    edges: List[Tuple[int, int]] = None,
    keypoint_names: Dict[int, str] = None,
    bbox: np.ndarray = None,
    boxes: bool = True,
    dpi: int = 200
) -> None:
    """
    Args:
        image (ndarray): [H, W, C]
        keypoints (ndarray): [N, 3]
        edges (List(Tuple(int, int))):
    """
    np.random.seed(42)
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range(len(keypoints))}

    if boxes:
        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 91), thickness=3)

    if bbox is not None:
        cv2.rectangle(image, tuple(bbox[0]), tuple(bbox[1]), (255, 100, 91), thickness=1)

    for i, keypoint in enumerate(keypoints):
        cv2.circle(image, tuple(keypoint), 1, colors.get(i), thickness=3, lineType=cv2.FILLED)

        if keypoint_names is not None:
            cv2.putText(
                image,
                f'{i}: {keypoint_names[i]}',
                tuple(keypoint),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    # cv2.circle(image, tuple(keypoints[0]), 3, colors.get(i), thickness=20, lineType=cv2.FILLED)

    if edges is not None:
        for i, edge in enumerate(edges):
            cv2.line(
                image,
                tuple(keypoints[edge[0]]),
                tuple(keypoints[edge[1]]),
                colors.get(edge[0]), 1, lineType=cv2.LINE_AA)

    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(image)
    # ax.axis('off')
    ax.axis('on')
    plt.show()
    fig.savefig('example.png')

keypoints = df_keypoints.loc[frame_num].values.reshape(-1, 2)
# print(keypoints.shape)
# print(keypoints)
for i in range(len(keypoints)):
    # keypoints[i][0] = keypoints[i][0] / 3.75      # 1920 / 512
    # keypoints[i][1] = keypoints[i][1] / 2.547     # 1080 / 424
    keypoints[i][0] = keypoints[i][0] / 5       # 1920 / 384
    keypoints[i][1] = keypoints[i][1] / 3.75    # 1080 / 288
# print(keypoints)
keypoints = keypoints.astype(np.int64)

bbox = df_bbox.loc[frame_num].values.reshape(-1, 2)
for i in range(len(bbox)):
    # bbox[i][0] = bbox[i][0] / 3.75      # 1920 / 512
    # bbox[i][1] = bbox[i][1] / 2.547     # 1080 / 424
    bbox[i][0] = bbox[i][0] / 5       # 1920 / 384
    bbox[i][1] = bbox[i][1] / 3.75    # 1080 / 288
bbox = bbox.astype(np.int64)
print(bbox[0])

keypoint_names = {
    0:'nose',                  
    1:'left_eye',              
    2:'right_eye',             
    3:'left_ear',              
    4:'right_ear',             
    5:'left_shoulder',         
    6:'right_shoulder',        
    7:'left_elbow',            
    8:'right_elbow',           
    9:'left_wrist',            
    10:'right_wrist',           
    11:'left_hip',                 
    12:'right_hip',                
    13:'left_knee',                
    14:'right_knee',               
    15:'left_ankle',               
    16:'right_ankle',        
}

edges = [
    (0, 1), (0, 2), (1, 3), (2, 4), (1, 2), (3, 5), (4, 6), # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), # 
    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16), # 
]

image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
# print(image.shape, type(image.shape))
# # h, w, c = image.shape
# print('width: ', image.shape[1])
# print('height:', image.shape[0])
# resize_img = cv2.resize(image, (512, 424))
# print('width: ', resize_img.shape[1])
# print('height:', resize_img.shape[0])
# https://daewoonginfo.blogspot.com/2019/05/opencv-python-resize.html
# draw_keypoints(image, keypoints, edges=edges, keypoint_names=keypoint_names, boxes=False, dpi=400)
draw_keypoints(image, keypoints, edges=edges, keypoint_names=keypoint_names, bbox=bbox, boxes=False, dpi=400)
# test_key = np.array([[189, 150], [189, 150]])
# custom_draw_keypoints(image, keypoints)
