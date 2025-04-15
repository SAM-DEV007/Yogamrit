import cv2
import copy
import time
import itertools
import csv
import ast

import mediapipe as mp
import tensorflow as tf
import numpy as np


from tensorflow.keras.models import load_model
from pathlib import Path

correction_folder = Path(__file__).resolve().parent / 'Correction_Data'

model_data = str(Path(__file__).resolve().parent / 'Model/model_v6.keras')
model = load_model(model_data)




mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(enable_segmentation=False, model_complexity=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)



ANGLE_THRESHOLD = 15
INCORRECT_ANGLE_THRESHOLD = 30
INCORRECT = 0
FRAME_COUNT = 0
MIN_INCORRECT = 3
MIN_THRESHOLD = 10
MAX_THRESHOLD = 30
THRESHOLD_MULTIPLIER = 2
DEBOUNCE_THRESHOLD_TIME = 1
DEBOUNCE_THRESHOLD = 0


_predict = None
PREDICT_THRESHOLD = 0.8
predictions = ['Hasta Uttanasan', 'Panchim Uttanasan', 'Vrikshasana', 'Vajrasana', 'Taadasana', 'Padmasana', 'Bhujangasana']
DEBOUNCE_TIME = 1
debounce = 0
prev_text = None
perform_detect = True


show_all_points = True

poses = ('NOSE', 'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX')
points_new_coll = [0 for _ in range(len(poses) + 1)]

data = {}
data_flip = {}
with open(str(correction_folder / f'data_correction_v3.csv'), 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        val = [ast.literal_eval(i) for i in row[1:-1]]

        if int(row[-1]) == 1:
            data_flip[row[0]] = val
            continue
        data[row[0]] = val




def calculate_angle(a, b, c):
    a = np.array(a) 
    b = np.array(b) 
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle


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

def process_frame(cap):
    global correction_folder, model_data, model, mp_pose, mp_drawing, pose, ANGLE_THRESHOLD, INCORRECT_ANGLE_THRESHOLD, INCORRECT, FRAME_COUNT, MIN_INCORRECT, MIN_THRESHOLD, MAX_THRESHOLD, THRESHOLD_MULTIPLIER, DEBOUNCE_THRESHOLD_TIME, DEBOUNCE_THRESHOLD, _predict, PREDICT_THRESHOLD, predictions, DEBOUNCE_TIME, debounce, prev_text, perform_detect, show_all_points, poses, points_new_coll, data, data_flip

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        print("Can't receive frame (Video end?). Exiting ...")
        exit()

    height, width, _ = frame.shape
    FRAME_COUNT += 1

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if (time.time() - debounce) > DEBOUNCE_TIME:
        debounce = time.time()
        perform_detect = True

    if results.pose_landmarks:
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            if mp_pose.PoseLandmark(idx).name in poses:
                index_pose = poses.index(mp_pose.PoseLandmark(idx).name)
                points_new_coll[index_pose] = np.array((landmark.x, landmark.y))
        points_new_coll[-1] = (np.array(((points_new_coll[9][0] + points_new_coll[10][0]) / 2, (points_new_coll[9][1] + points_new_coll[10][1]) / 2)))

        points_new_coll = np.array(points_new_coll)

        if perform_detect:
            perform_detect = False

            norm = landmark_list(frame, points_new_coll)
            predict_model = np.squeeze(model.predict(np.array([norm]), verbose=0))
            prediction = np.argmax(predict_model)
            
            if predict_model[prediction] >= PREDICT_THRESHOLD:
                _predict = prediction

            if _predict:
                prev_text = predictions[_predict]

        if prev_text:
            points_new = points_new_coll[1:-1]
            points = data[prev_text][1:-1]
            points_flip = data_flip[prev_text][1:-1]

            non_flip, flip = 0, 0
            new, ref, ref_flip = [], [], []
            for i in range(2, len(points_new) - 2):
                av = 2
                if i == 6 or i == 8:
                    av = 1

                angle_new = round(calculate_angle(points_new[i - 2], points_new[i], points_new[i + av]))
                angle_ref = round(calculate_angle(points[i - 2], points[i], points[i + av]))
                angle_ref_flip = round(calculate_angle(points_flip[i - 2], points_flip[i], points_flip[i + av]))

                new.append(angle_new)
                ref.append(angle_ref)
                ref_flip.append(angle_ref_flip)

                if abs(angle_new - angle_ref) > ANGLE_THRESHOLD:
                    non_flip += abs(angle_new - angle_ref)
                if abs(angle_new - angle_ref_flip) > ANGLE_THRESHOLD:
                    flip += abs(angle_new - angle_ref_flip)
            
            use_flip = False
            if non_flip > flip: # Use the one with the least error
                use_flip = True

            for i in range(2, len(points_new) - 2):
                av = 2
                if i == 6 or i == 8:
                    av = 1

                clr = (0, 255, 0) # Green
                
                

                angle_new = new[i - 2]
                angle = (ref[i - 2], ref_flip[i - 2])[use_flip]

                if show_all_points:
                    frame = cv2.circle(frame, (int(points_new[i - 2][0] * width), int(points_new[i - 2][1] * height)), 4, clr, -1)
                    frame = cv2.circle(frame, (int(points_new[i + av][0] * width), int(points_new[i + av][1] * height)), 4, clr, -1)

                    frame = cv2.line(frame, (int(points_new[i][0] * width), int(points_new[i][1] * height)), (int(points_new[i + av][0] * width), int(points_new[i + av][1] * height)), clr, 1)

                
                if abs(angle_new - angle) > ANGLE_THRESHOLD:
                    min_dev = abs(angle_new - angle)

                    if abs(ANGLE_THRESHOLD - min_dev) > INCORRECT_ANGLE_THRESHOLD:
                        clr = (0, 0, 255) # Red
                    else:
                        clr = (0, 255, 255) # Yellow

                    frame = cv2.circle(frame, (int(points_new[i - 2][0] * width), int(points_new[i - 2][1] * height)), 4, clr, -1)
                    frame = cv2.circle(frame, (int(points_new[i + av][0] * width), int(points_new[i + av][1] * height)), 4, clr, -1)

                    frame = cv2.line(frame, (int(points_new[i][0] * width), int(points_new[i][1] * height)), (int(points_new[i + av][0] * width), int(points_new[i + av][1] * height)), clr, 1)
                    

                if show_all_points:
                    frame = cv2.line(frame, (int(points_new[i][0] * width), int(points_new[i][1] * height)), (int(points_new[i - 2][0] * width), int(points_new[i - 2][1] * height)), clr, 1)
                    frame = cv2.circle(frame, (int(points_new[i][0] * width), int(points_new[i][1] * height)), 4, clr, -1)

                #if abs(angle_new - angle_ref) > ANGLE_THRESHOLD and abs(angle_new - angle_ref_flip) > ANGLE_THRESHOLD:
                if abs(angle_new - angle) > ANGLE_THRESHOLD:
                    frame = cv2.line(frame, (int(points_new[i][0] * width), int(points_new[i][1] * height)), (int(points_new[i - 2][0] * width), int(points_new[i - 2][1] * height)), clr, 1)
                    frame = cv2.circle(frame, (int(points_new[i][0] * width), int(points_new[i][1] * height)), 4, clr, -1)

                    INCORRECT += 1

                    
    
    if (time.time() - DEBOUNCE_THRESHOLD) > DEBOUNCE_THRESHOLD_TIME:
        actual_incorrect = INCORRECT / FRAME_COUNT
        if actual_incorrect > MIN_INCORRECT:
            ANGLE_THRESHOLD = min(MAX_THRESHOLD, ANGLE_THRESHOLD + THRESHOLD_MULTIPLIER)
        else:
            ANGLE_THRESHOLD = max(MIN_THRESHOLD, ANGLE_THRESHOLD - THRESHOLD_MULTIPLIER)
        
        DEBOUNCE_THRESHOLD = time.time()
        INCORRECT = 0
        FRAME_COUNT = 0



    '''if prev_text:
        cv2.putText(frame, prev_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)'''
    

    return frame,prev_text


if __name__ == "__main__":
    process_frame()