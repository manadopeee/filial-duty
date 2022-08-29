from genericpath import isdir
import os
import glob
import shutil
from zipfile import ZipFile

label_dict = { 1 : '수저 또는 포크로 음식 집어먹기', 2 : '물 또는 음료를 컵에 따르기', 3 : '약 먹기', 4 : '물 또는 음료 마시기', 5 : '냉장고에 음식 넣고 꺼내기', 6 : '채소 다듬기', 7 : '과일 깎기', 8 : '가스레인지로 음식 데우기', 9 : '칼로 도마 위의 음식 자르기', 10 : '이빨 닦기',
             11 : '손 씻기', 12 : '세수하기', 13 : '수건으로 얼굴/머리 닦기', 14 : '화장품 바르기', 15 : '립스틱 바르기', 16 : '머리 빗기', 17 : '머리 드라이기로 말리기', 18 : '상의 입기', 19 : '상의 벗기', 20 : '신발 신고 벗기',
             21 : '안경 쓰고 벗기', 22 : '설거지하기', 23 : '진공청소기 사용하기', 24 : '걸레로 엎드려서 바닥 닦기', 25 : '식탁을 행주로 닦기', 26 : '창문이나 가구 등 닦기', 27 : '이불 펴고 개기', 28 : '손빨래 하기', 29 : '빨래 널기', 30 : '물건을 찾기 위해 두리번거리기',
             31 : '리모컨으로 TV 컨트롤하기', 32 : '책 읽기', 33 : '신문 보기', 34 : '글쓰기', 35 : '전화 걸거나 받기', 36 : '스마트폰 조작하기', 37 : '컴퓨터 키보드 치기', 38 : '담배 피기', 39 : '박수 치기', 40 : '두 손으로 얼굴 비비기',
             41 : '맨손체조 하기', 42 : '목 돌리기 운동 하기', 43 : '어깨 셀프 안마 하기', 44 : '고개 숙여 인사 하기', 45 : '담소 나누기', 46 : '악수 하기', 47 : '포옹 하기', 48 : '서로 싸우기', 49 : '손을 좌우로 흔들기 (waving)', 50 : '이리 오라고 손짓하기 (calling)',
             51 : '손가락으로 가리키기 (pointing)', 52 : '문을 열고 들어가기', 53 : '쓰러지기', 54 : '누워있다 일어나기', 55 : '서 있다가 눕기' }

essential_label = {3:'take_medicine', 31:'tv_remote_control', 53:'fall_down'}
target_label = []
for label in essential_label:
    label = str(label)
    target_label.append('A' + label.zfill(3))

RGB_list = [
    # 'RGB_P001-P010', 'RGB_P011-P020', 'RGB_P021-P030', 'RGB_P031-P040', 'RGB_P041-P050', 'RGB_P051-P060', 'RGB_P061-P070', 
    'RGB_P071-P080', 'RGB_P081-P090', 'RGB_P091-P100']

# ZipFile('/home/minu/다운로드/hyodol/RGB_P011-P020.zip').extractall()

for rgb_list in RGB_list:
    zip_name = f'/home/minu/다운로드/hyodol/{rgb_list}.zip'
    with ZipFile(zip_name, 'r') as zip:
        print(f'{rgb_list}.zip')
        # zip.printdir()
        zip.extractall('/home/minu/다운로드/hyodol/')
        zip.close()
    print(f'unzip done...')

    people_folder = rgb_list.split('_')[-1]
    VIDEO_PATHS = f'/home/minu/다운로드/hyodol/{people_folder}'
    OUTPUT_PATH = f'/home/minu/다운로드/hyodol/essential_label'
    video_path_list = glob.glob(VIDEO_PATHS + "/*/*.mp4")

    for video_path in video_path_list:
        video_name = video_path.split('/')[-1]
        people_num = video_path.split('/')[-3:-1]
        output_path = os.path.join(OUTPUT_PATH, people_num[0], people_num[1])

        for label in target_label:
            if label in video_name:
                if not os.path.isdir(output_path):
                    os.makedirs(output_path)
                shutil.copyfile(video_path, os.path.join(output_path, video_name))
    
    if os.path.exists(zip_name):
        os.remove(zip_name)
        print(f'remove {rgb_list}.zip')
    if os.path.exists(VIDEO_PATHS):
        shutil.rmtree(VIDEO_PATHS)
        print(f'rmtree {people_folder}')
    print(people_folder, 'done...')
print('done!')
