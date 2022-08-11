# https://ponyozzang.tistory.com/439
import os
import glob
import shutil

image_path_root = 'D:/DATASET/hyodol/OUTPUT'
image_paths = glob.glob(image_path_root + '/*/*/*.jpg')
out_dir = 'D:/DATASET/hyodol/OUTPUT_imgs'

try:
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
except OSError:
    print('Error: Creating directory. ' + out_dir)

for i, image_path in enumerate(image_paths):
    shutil.copy2(image_path, out_dir)
    print(f'{i+1} / {len(image_paths)}')
print('done')