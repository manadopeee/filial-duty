# https://docs.opencv.org/3.4/d4/d15/group__videoio__flags__base.html
#라이브러리 호출
import cv2
import os
import glob

# print(cv2.__version__)

# A0xx = 행동(action)
# P0xx = 사람(people)
# G00x = 장소(g1 kitchen, g2 living room, g3 bedroom)
# C00x = 카메라 위치, 각도(camera)
today = '2022.07.26'
flag = 'Action' # People

label_dict = { 1 : '수저 또는 포크로 음식 집어먹기', 2 : '물 또는 음료를 컵에 따르기', 3 : '약 먹기', 4 : '물 또는 음료 마시기', 5 : '냉장고에 음식 넣고 꺼내기', 6 : '채소 다듬기', 7 : '과일 깎기', 8 : '가스레인지로 음식 데우기', 9 : '칼로 도마 위의 음식 자르기', 10 : '이빨 닦기',
             11 : '손 씻기', 12 : '세수하기', 13 : '수건으로 얼굴/머리 닦기', 14 : '화장품 바르기', 15 : '립스틱 바르기', 16 : '머리 빗기', 17 : '머리 드라이기로 말리기', 18 : '상의 입기', 19 : '상의 벗기', 20 : '신발 신고 벗기',
             21 : '안경 쓰고 벗기', 22 : '설거지하기', 23 : '진공청소기 사용하기', 24 : '걸레로 엎드려서 바닥 닦기', 25 : '식탁을 행주로 닦기', 26 : '창문이나 가구 등 닦기', 27 : '이불 펴고 개기', 28 : '손빨래 하기', 29 : '빨래 널기', 30 : '물건을 찾기 위해 두리번거리기',
             31 : '리모컨으로 TV 컨트롤하기', 32 : '책 읽기', 33 : '신문 보기', 34 : '글쓰기', 35 : '전화 걸거나 받기', 36 : '스마트폰 조작하기', 37 : '컴퓨터 키보드 치기', 38 : '담배 피기', 39 : '박수 치기', 40 : '두 손으로 얼굴 비비기',
             41 : '맨손체조 하기', 42 : '목 돌리기 운동 하기', 43 : '어깨 셀프 안마 하기', 44 : '고개 숙여 인사 하기', 45 : '담소 나누기', 46 : '악수 하기', 47 : '포옹 하기', 48 : '서로 싸우기', 49 : '손을 좌우로 흔들기 (waving)', 50 : '이리 오라고 손짓하기 (calling)',
             51 : '손가락으로 가리키기 (pointing)', 52 : '문을 열고 들어가기', 53 : '쓰러지기', 54 : '누워있다 일어나기', 55 : '서 있다가 눕기' }
label_dict_eng = {3:'take_medicine', 12:'wash_face', 30:'find_things', 41:'gymnastics', 53:'fall_down'}
essential_label = {3:'take_medicine', 31:'tv_remote_control', 53:'fall_down'}
VIDEO_PATHS = 'D:/DATASET/hyodol/P001-P010'
OUTPUT_PATH = f'D:/DATASET/hyodol/OUTPUT_images_for_{flag}_{today}'
videos = glob.glob(VIDEO_PATHS + "/*/*.mp4")
# print(videos)
# filepath = 'D:/DATASET/hyodol/test/A001_P001_G001_C001.mp4'
# video_path = 'D:/DATASET/hyodol/test/A001_P001_G001_C001.mp4'
# target_label = [3, 12, 30, 41, 53]
target_label = essential_label

for i, video_path in enumerate(videos):
    label = video_path.split('\\')[-1].split('_')[0][1:]
    # if video_path.split('\\')[-1].split('_')[0][-1] == str(i+1):  # 3이 약먹기
    if (int(label)) in target_label:
        video_name = video_path.split('\\')[-1][:-4]
        people_name = video_name.split('_')[1]
        if flag == 'People':
            out_dir = OUTPUT_PATH + '/' + target_label[int(label)] + '/' + video_name # 사람 마다
        else:
            out_dir = OUTPUT_PATH + '/' + target_label[int(label)] # 행동 마다
        video = cv2.VideoCapture(video_path) #'' 사이에 사용할 비디오 파일의 경로 및 이름을 넣어주도록 함

        if not video.isOpened():
            print("Could not Open :", video_path)
            exit(0)

        #불러온 비디오 파일의 정보 출력
        length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)

        # print("length :", length)
        # print("width :", width)
        # print("height :", height)
        # print("fps :", fps)

        #프레임을 저장할 디렉토리를 생성
        try:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
        except OSError:
            print('Error: Creating directory. ' + out_dir)

        count = 1

        while (video.isOpened()):
            ret, image = video.read()
            if (int(video.get(1)) % int(fps/fps) == 0):  # 앞서 불러온 fps 값을 사용하여 1초마다 추출, fps를 낮출 수록 전체 프레임 저장
                resize_img = cv2.resize(image, (512, 424)) # resize for kinnect v2 size
                cv2.imwrite(out_dir + f"/{video_name}_frame_%d.jpg" % count, resize_img)
                # print('Saved frame number :', str(int(video.get(1))))
                count += 1
            if int(video.get(1)) == length:
                break
        video.release()
        print(f'{people_name} > {target_label[int(label)]}\t >> {i+1} / {len(videos)}')
print('done!')

# 성능이 안좋을 시 1초당 1장이 아니라 더 이미지 받아야 될듯...
# 레이어 변경 해보고, k-fold 적용해보고...