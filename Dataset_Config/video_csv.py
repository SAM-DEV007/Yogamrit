import cv2
import copy
import itertools
import os

import mediapipe as mp
import numpy as np
import pandas as pd

from pathlib import Path
from csv import writer


def write_to_csv(file, data, label):
    with open(file, 'a', newline='') as f:
        wo = writer(f)
        wo.writerow([label] + data)


def preprocess_data(landmark_list):
    temp_list = copy.deepcopy(landmark_list)

    base_x, base_y = 0, 0
    for idx, lp in enumerate(temp_list):
        if idx == 0:
            base_x, base_y = lp[0], lp[1]
            
        temp_list[idx][0] -= base_x
        temp_list[idx][1] -= base_y
    
    temp_list = list(itertools.chain.from_iterable(temp_list))
    max_val = max(list(map(abs, temp_list)))

    def normalize_(n):
        return n / max_val

    temp_list = list(map(normalize_, temp_list))

    return temp_list


def landmark_list(frame, pose):
    height, width, _ = frame.shape

    landmarks = []

    for landmark in pose:
        lx = min(int(landmark[0] * width), width - 1)
        ly = min(int(landmark[1] * height), height - 1)

        landmarks.append([lx, ly])
    
    return preprocess_data(landmarks)


if __name__ == "__main__":
    video_folder = Path(__file__).resolve().parent / 'Video'
    file_dir = Path(__file__).resolve().parent / 'Data'

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    v1_1 = str(video_folder / 'Hasta Uttanasan/1_HU.mp4')
    v1_2 = str(video_folder / 'Hasta Uttanasan/2_HU.mp4')

    v2_1 = str(video_folder / 'Panchim Uttanasan/1_PU.mp4')

    v3_1 = str(video_folder / 'Vrikshasana/1_V.mp4')
    v3_2 = str(video_folder / 'Vrikshasana/Test2.mp4')
    v3_3 = str(video_folder / 'Vrikshasana/2_V.mp4')
    v3_4 = str(video_folder / 'Vrikshasana/3_V.mp4')

    v4_1 = str(video_folder / 'Vajrasana/1.mp4')
    v4_2 = str(video_folder / 'Vajrasana/2.mp4')
    v4_3 = str(video_folder / 'Vajrasana/3.mp4')

    v5_1 = str(video_folder / 'Taadasana/1.mp4')
    v5_2 = str(video_folder / 'Taadasana/2.mp4')
    v5_3 = str(video_folder / 'Taadasana/3.mp4')

    v6_1 = str(video_folder / 'Padmasana/1.mp4')
    v6_2 = str(video_folder / 'Padmasana/2.mp4')

    v7_1 = str(video_folder / 'Bhujangasana/1_B.mp4')
    v7_2 = str(video_folder / 'Bhujangasana/2_B.mp4')
    v7_3 = str(video_folder / 'Bhujangasana/3_B.mp4')

    v8_1 = str(video_folder / 'NoAsana/1.mp4')
    v8_2 = str(video_folder / 'NoAsana/2.mp4')
    v8_3 = str(video_folder / 'NoAsana/3.mp4')
    v8_4 = str(video_folder / 'NoAsana/4.mp4')
    v8_5 = str(video_folder / 'NoAsana/5.mp4')
    v8_6 = str(video_folder / 'NoAsana/6.mp4')
    v8_7 = str(video_folder / 'NoAsana/7.mp4')
    v8_8 = str(video_folder / 'NoAsana/8.mp4')
    v8_9 = str(video_folder / 'NoAsana/9.mp4')

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(enable_segmentation=False, model_complexity=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)

    poses = ('NOSE', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX')
    points_new_coll = [0 for _ in range(len(poses) + 1)]

    vids = [(v1_1,), (v2_1,), (v3_1, v3_2, v3_3, v3_4), (v4_1, v4_2, v4_3), (v5_1, v5_2, v5_3), (v6_1, v6_2), (v7_1, v7_2, v7_3), (v8_1, v8_2, v8_3)]

    new_vids = {0: (v1_2,), 7: (v8_4, v8_5, v8_6, v8_7, v8_8, v8_9)}

    SKIP = False
    SKIP_CLASS = 7  # 0: Hasta Uttanasan, 1: Panchim Uttanasan, 2: Vrikshasana, 3: Vajrasana, 4: Taadasana, 5: Padmasana, 6: Bhujangasana, 7: NoAsana
    #for i, v_list in enumerate(vids):
    for i, v_list in new_vids.items():
        if SKIP and i < SKIP_CLASS:
            continue
        print(f"Processing video {i + 1} ...")
        for v in v_list:
            for j in range(2):
                if (j == 0):
                    print("Processing video in mirror mode ...")
                else:
                    print("Processing video in normal mode ...")

                cap = cv2.VideoCapture(v)

                while cap.isOpened():
                    ret, frame = cap.read()

                    if (j == 0):
                        frame = cv2.flip(frame, 1)

                    if not ret:
                        print("Can't receive frame (Video end?). Exiting ...")
                        break

                    # frame = cv2.resize(frame, (700, 700))
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(frame_rgb)

                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            if mp_pose.PoseLandmark(idx).name in poses:
                                index_pose = poses.index(mp_pose.PoseLandmark(idx).name)
                                points_new_coll[index_pose] = np.array((landmark.x, landmark.y))
                        points_new_coll[-1] = (np.array(((points_new_coll[9][0] + points_new_coll[10][0]) / 2, (points_new_coll[9][1] + points_new_coll[10][1]) / 2)))

                        data = landmark_list(frame, points_new_coll)
                        write_to_csv(str(file_dir / 'data_v9.csv'), data, i)

                    cv2.imshow('Video', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    if cv2.waitKey(1) == 27:
                        break

                    if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
                        break

    cap.release()
    cv2.destroyAllWindows()