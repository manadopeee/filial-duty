import json, os
import numpy as np
import pandas as pd
import glob
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from itertools import zip_longest

csv_dir = './data/hyodol/threshold/keypoint'
csv_path = sorted(glob.glob(csv_dir + "/*/*"))
people = 1
height = 384
width = 288
threshold = 0.8
div_x = 1920 / 384
div_y = 1080 / 288

train_csv = csv_path[:int(len(csv_path) * threshold)]
valid_csv = csv_path[int(len(csv_path) * threshold):]
# print(len(train_csv), len(valid_csv))
train_csv_list = []
valid_csv_list = []

for csv in zip_longest(train_csv, valid_csv):
    train_csv_list.extend(glob.glob(csv[0] + '/*.csv'))
    if csv[1] != None:
        valid_csv_list.extend(glob.glob(csv[1] + '/*.csv'))

for list_type, csv_file in enumerate([train_csv_list, valid_csv_list]):
    file_data_all = OrderedDict()
    file_data_all['img_ann'] = []

    for csv_path in tqdm(csv_file):
        file_name = csv_path.split('/')[-1].split('.')[0]
        path = '/'.join(csv_path.split('/')[5:-1])
        img_id = ''
        for i, fn in enumerate(file_name.split('_')):
            img_id += fn[1:]
        df_csv = pd.read_csv(csv_path)

        #keypoint
        df_keypoint_column = []
        column_x = []
        column_y = []
        for column in df_csv:
            if '_x' in column:
                df_keypoint_column.append(column)
                column_x.append(column)
            elif '_y' in column:
                df_keypoint_column.append(column)
                column_y.append(column)
        df_keypoints = df_csv[df_keypoint_column]
        df_keypoints.update(df_keypoints[column_x].div(div_x), overwrite=True)
        df_keypoints.update(df_keypoints[column_y].div(div_y), overwrite=True)

        # bbox
        df_bbox_column = []
        for column in df_csv:
            if 'bbox' in column and 'score' not in column:
                df_bbox_column.append(column)
        df_bbox = df_csv[df_bbox_column]
        df_bbox.update(df_bbox.iloc[:, [0, 1]].div(div_x))
        df_bbox.update(df_bbox.iloc[:, [2, 3]].div(div_y))

        for image_num in range(len(df_csv)):      
            #keypoint 
            keypoint = df_keypoints.loc[image_num].values.tolist()
            # bbox
            bbox = df_bbox.loc[image_num].values.tolist()
            fill_zero = str(image_num + 1).zfill(3)
            img_id_num = img_id + fill_zero
            
            file_data_all['img_ann'].append({'file_name': f'{path}/{file_name}_frame_{fill_zero}.jpg', 'height': height, 'width': width, 
                                            'keypoints': keypoint,
                                            'bbox': bbox})

    if list_type == 0:
        with open(f'./data/hyodol/json_results/keypoints_train.json', 'w', encoding='utf-8') as make_file:
            json.dump(file_data_all, make_file, ensure_ascii=False, indent='\t')
        print('keypoints_train.json save done...')
    else:
        with open(f'./data/hyodol/json_results/keypoints_valid.json', 'w', encoding='utf-8') as make_file:
            json.dump(file_data_all, make_file, ensure_ascii=False, indent='\t')
        print('keypoints_valid.json save done...')
