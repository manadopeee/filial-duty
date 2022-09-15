import pandas as pd
import os
import glob

csv_path = './data/hyodol/xy_results'

keypoint_csv_list = glob.glob(csv_path + "/keypoint/*/*/*")

pass_num = 1
bbox_threshold = 0.9
keypoint_threshold = 0.70

for csv_num, keypoint_csv in enumerate(keypoint_csv_list):
    read_csv_df = pd.read_csv(keypoint_csv)
    keypoint_df = read_csv_df.iloc[:, 5:]
    mean_bbox_score = read_csv_df['bbox_score'].mean()
    keypoint_score = pd.DataFrame()

    path_split = keypoint_csv.split('/')
    csv_name = path_split[-1]
    save_path = os.path.join('/'.join(path_split[:3]), 'threshold', '/'.join(path_split[4:-1]))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # bbox mean이 threshold 이상이고 keypoint mean도 threshold이상인 csv
    if mean_bbox_score > bbox_threshold:
        for i in range(int(len(keypoint_df.columns) / 3)):
            keypoint_score = pd.concat([keypoint_score, keypoint_df.iloc[:, 2+i*3]], axis=1) # 2, 5, 8
        keypoint_mean = keypoint_score.mean()
        # keypoint_total_mean = keypoint_score.mean().describe()['mean']

        if keypoint_mean.min() < keypoint_threshold:
            continue
        else:
            read_csv_df.to_csv(f'{save_path}/{csv_name}', index=False)
            pass_num += 1
        print(f'{pass_num}... {csv_num} / {len(keypoint_csv_list)}, path : {save_path}/{csv_name}')
