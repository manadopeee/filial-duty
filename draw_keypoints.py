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

import albumentations as A
from albumentations.pytorch import ToTensorV2

label = 'take_medicine'
file_name = 'A003_P001_G001_C008_frame_1'
csv_name = file_name.split('_frame')[:-1][0]
frame_num = int(file_name.split('_')[-1])
print(f'csv_name : {csv_name}, frame_num : {frame_num}')
csv_path = f'D:/DATASET/hyodol/Skeleton(P001-P100)/P001-P050/{csv_name}.csv'
df = pd.read_csv(csv_path, index_col='frameNum')
df.head()

# df_keypoints = df[]
# df_column = list(df)
df_column = []
for column in df:
    if 'depth' in column:
        df_column.append(column)

df_keypoints = df[df_column]
df_keypoints.loc[1].shape
df_keypoints.loc[frame_num]

def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    edges: List[Tuple[int, int]] = None,
    keypoint_names: Dict[int, str] = None,
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
    colors = {k: tuple(map(int, np.random.randint(0, 255, 3))) for k in range(25)}

    if boxes:
        x1, y1 = min(keypoints[:, 0]), min(keypoints[:, 1])
        x2, y2 = max(keypoints[:, 0]), max(keypoints[:, 1])
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 100, 91), thickness=3)

    for i, keypoint in enumerate(keypoints):
        cv2.circle(image, tuple(keypoint), 3, colors.get(i), thickness=3, lineType=cv2.FILLED)

        if keypoint_names is not None:
            cv2.putText(
                image,
                f'{i}: {keypoint_names[i]}',
                tuple(keypoint),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    # cv2.circle(image, tuple(keypoints[0]), 3, colors.get(i), thickness=20, lineType=cv2.FILLED)

    if edges is not None:
        for i, edge in enumerate(edges):
            cv2.line(
                image,
                tuple(keypoints[edge[0]]),
                tuple(keypoints[edge[1]]),
                colors.get(edge[0]), 2, lineType=cv2.LINE_AA)

    fig, ax = plt.subplots(dpi=dpi)
    ax.imshow(image)
    # ax.axis('off')
    ax.axis('on')
    plt.show()
    fig.savefig('example.png')

keypoints = df_keypoints.loc[frame_num].values.reshape(-1, 2)
# print(keypoints)
keypoints = keypoints.astype(np.int64)
keypoint_names = {
    0: 'SpineBase',
    1: 'SpineMid',
    2: 'Neck',
    3: 'Head',
    4: 'ShoulderLeft',
    5: 'ElbowLeft',
    6: 'WristLeft',
    7: 'HandLeft',
    8: 'ShoulderRight',
    9: 'ElbowRight',
    10:'WristRight',
    11:'HandRight',
    12:'HipLeft',
    13:'KneeLeft',
    14:'AnkleLeft',
    15:'FootLeft ',
    16:'HipRight',
    17:'KneeRight',
    18:'AnkleRight',
    19:'FootRight',
    20:'SpineShoulder',
    21:'HandTipLeft',
    22:'ThumbLeft',
    23:'HandTipRight',
    24:'ThumbRight',
}

edges = [
    (0, 1), (1, 20), (20, 2), (2, 3), (20, 4), (4, 5), (5, 6), (6, 7), (20, 8), (8, 9), (9, 10), (10, 11), # 12 상체
    (7, 21), (7, 22), (11, 23), (11, 24), # 4 손
    (0, 12), (12, 13), (13, 14), (14, 15), (0, 16), (16, 17), (17, 18), (18, 19), # 8 하체
]
# img_path = f'D:/DATASET/hyodol/OUTPUT_images/{label}/{file_name}.jpg'
img_path = f'D:/DATASET/hyodol/OUTPUT_images_for_Action_2022.07.26/{label}/{file_name}.jpg'
print('img_path : ', img_path)
print('csv_path : ', csv_path)
image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
# print(image.shape, type(image.shape))
# # h, w, c = image.shape
# print('width: ', image.shape[1])
# print('height:', image.shape[0])
# resize_img = cv2.resize(image, (512, 424))
# print('width: ', resize_img.shape[1])
# print('height:', resize_img.shape[0])
# https://daewoonginfo.blogspot.com/2019/05/opencv-python-resize.html
draw_keypoints(image, keypoints, edges=edges, keypoint_names=keypoint_names, boxes=False, dpi=400)
# test_key = np.array([[189, 150], [189, 150]])
# custom_draw_keypoints(image, keypoints)