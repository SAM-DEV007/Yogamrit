from pathlib import Path

import os
import cv2
# import pickle
import csv

import mediapipe as mp
import numpy as np


if __name__ == '__main__':
    SAVE = True
    image_folder = Path(__file__).resolve().parent / 'Images'
    correction_folder = Path(__file__).resolve().parent / 'Correction_Data'

    i1 = str(image_folder / 'Vrikshasana/1.jpg')
    i2 = str(image_folder / 'Vrikshasana/2.jpg')
    i3 = str(image_folder / 'Vrikshasana/3.jpg')
    i5 = str(image_folder / 'Vrikshasana/5.jpg')

    v1 = str(image_folder / 'Hasta Uttanasan/1.jpg')
    v2 = str(image_folder / 'Panchim Uttanasan/1.jpg')
    v3 = str(image_folder / 'Vajrasana/1.jpg')
    v5 = str(image_folder / 'Taadasana/1.jpg')
    v6 = str(image_folder / 'Padmasana/1.jpg')
    v7 = str(image_folder / 'Bhujangasana/1.jpg')

    poses = ('NOSE', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX')
    _all = []

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(enable_segmentation=False, model_complexity=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)

    for i in [i5, v1, v2, v3, v5, v6, v7]:
        img = cv2.imread(i)
        _copy = False

        for j in range(2):
            points = []

        # if not SAVE:
            # with open(str(image_folder / 'Vrikshasana/1.pkl'), 'rb') as f:
                #   points = pickle.load(f)

            if j == 1:
                img = cv2.flip(img, 1)

            #img = cv2.resize(img, (700, 700))

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)
            cv2.imshow('Image', img)
            if results.pose_landmarks:
                #mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                #for idx, landmark in enumerate(results.pose_landmarks.landmark):
                #    print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}")

                if SAVE:
                    for landmark_name in poses:
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            if mp_pose.PoseLandmark(idx).name == landmark_name:
                                points.append((landmark.x, landmark.y))
                                #img = cv2.circle(img, (int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])), 4, (0, 255, 0), -1)

                    points.append(((points[7][0] + points[8][0]) / 2, (points[7][1] + points[8][1]) / 2))

                for point in points:
                    img = cv2.circle(img, (int(point[0] * img.shape[1]), int(point[1] * img.shape[0])), 4, (0, 255, 0), -1) 
                
                if SAVE:
                    '''
                    with open(str(correction_folder / f'Vrikshasana.pkl'), 'wb') as f:
                        pickle.dump(points, f)
                    '''
                    filename = i.split('\\')[-2]
                    print(f'Saving {filename} in {("Normal", "Flipped")[j == 1]} mode..., {j}')

                    if _copy:
                        _all.append([i.split('\\')[-2]] + points + [j-1])
                        _copy = False
                    _all.append([i.split('\\')[-2]] + points + [j])
                    continue
            
            if SAVE:
                if j == 1:
                    points = _all[-1][1:-1]
                    _all.append([i.split('\\')[-2]] + points + [j])
                else:
                    _copy = True

    with open(str(correction_folder / f'data_correction_v3.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(_all)
                
    '''with open(str(correction_folder / f'data_correction.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)'''
    
    import ast
    data = {}
    with open(str(correction_folder / f'data_correction_v3.csv'), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data[row[0]] = [ast.literal_eval(i) for i in row[1:]]
    
    #print(data)

    #cv2.imshow('Image', img)
    #cv2.waitKey(0)