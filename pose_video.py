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


if __name__ == "__main__":
    correction_folder = Path(__file__).resolve().parent / 'Correction_Data'
    video_folder = Path(__file__).resolve().parent / 'Tests/Video'

    v1 = str(video_folder / 'Hasta Uttanasan/1_HU.mp4')
    v2 = str(video_folder / 'Panchim Uttanasan/1_PU.mp4')
    v3 = str(video_folder / 'Vrikshasana/1_V.mp4')
    v4 = str(video_folder / 'Vajrasana/1.mp4')
    v5 = str(video_folder / 'Tadasana/1.mp4')
    v6 = str(video_folder / 'Padmasana/1.mp4')

    t3_1 = str(video_folder / 'Vrikshasana/Test1.mp4')
    t3_2 = str(video_folder / 'Vrikshasana/Test2.mp4')

    t7 = str(video_folder / 'Bhujangasana/1_B.mp4')

    model_data = str(Path(__file__).resolve().parent / 'Model/model_v5.keras')
    model = load_model(model_data)

    #cap = cv2.VideoCapture(t3_2)
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 700)
    #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 900)

    cv2.namedWindow('Video', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(enable_segmentation=False, model_complexity=1, min_detection_confidence=0.3, min_tracking_confidence=0.3)


    ANGLE_THRESHOLD = 10
    INCORRECT = 0
    MIN_INCORRECT = 10
    MIN_THRESHOLD = 10
    MAX_THRESHOLD = 30
    THRESHOLD_MULTIPLIER = 2
    DEBOUNCE_THRESHOLD_TIME = 1
    DEBOUNCE_THRESHOLD = 0

    predictions = ['Hasta Uttanasan', 'Panchim Uttanasan', 'Vrikshasana', 'Vajrasana', 'Taadasana', 'Padmasana', 'Bhujangasana']
    DEBOUNCE_TIME = 2
    debounce = 0
    prev_text = None
    perform_detect = True

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

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Can't receive frame (Video end?). Exiting ...")
            break

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

            if perform_detect:
                perform_detect = False

                norm = landmark_list(frame, points_new_coll)
                _predict = model.predict(np.array([norm]))
                _predict = np.argmax(np.squeeze(_predict))

                prev_text = predictions[_predict]

            points_new = points_new_coll[1:-1]
            points = data[prev_text][1:-1]
            points_flip = data_flip[prev_text][1:-1]
    
            for i in range(2, len(points_new) - 2):
                clr = (255, 0, 0)
                
                angle_new = round(calculate_angle(points_new[i - 2], points_new[i], points_new[i + 2]))
                angle_ref = round(calculate_angle(points[i - 2], points[i], points[i + 2]))
                angle_ref_flip = round(calculate_angle(points_flip[i - 2], points_flip[i], points_flip[i + 2]))

                if abs(angle_new - angle_ref) > ANGLE_THRESHOLD and abs(angle_new - angle_ref_flip) > ANGLE_THRESHOLD:
                    clr = (0, 0, 255)

                    frame = cv2.circle(frame, (int(points_new[i - 2][0] * frame.shape[1]), int(points_new[i - 2][1] * frame.shape[0])), 4, clr, -1)
                    frame = cv2.circle(frame, (int(points_new[i + 2][0] * frame.shape[1]), int(points_new[i + 2][1] * frame.shape[0])), 4, clr, -1)

                    frame = cv2.line(frame, (int(points_new[i][0] * frame.shape[1]), int(points_new[i][1] * frame.shape[0])), (int(points_new[i + 2][0] * frame.shape[1]), int(points_new[i + 2][1] * frame.shape[0])), clr, 1)
                if i == 6 or i == 8:            
                    angle_new = calculate_angle(points_new[i - 2], points_new[i], points_new[i + 1])
                    angle_ref = calculate_angle(points[i - 2], points[i], points[i + 1])

                    if abs(angle_new - angle_ref) > ANGLE_THRESHOLD and abs(angle_new - angle_ref_flip) > ANGLE_THRESHOLD:
                        clr = (0, 0, 255)

                        frame = cv2.circle(frame, (int(points_new[i - 2][0] * frame.shape[1]), int(points_new[i - 2][1] * frame.shape[0])), 4, clr, -1)
                        frame = cv2.circle(frame, (int(points_new[i + 1][0] * frame.shape[1]), int(points_new[i + 1][1] * frame.shape[0])), 4, clr, -1)
                        
                        frame = cv2.line(frame, (int(points_new[i][0] * frame.shape[1]), int(points_new[i][1] * frame.shape[0])), (int(points_new[i + 1][0] * frame.shape[1]), int(points_new[i + 1][1] * frame.shape[0])), clr, 1)

                if abs(angle_new - angle_ref) > ANGLE_THRESHOLD and abs(angle_new - angle_ref_flip) > ANGLE_THRESHOLD:
                    frame = cv2.line(frame, (int(points_new[i][0] * frame.shape[1]), int(points_new[i][1] * frame.shape[0])), (int(points_new[i - 2][0] * frame.shape[1]), int(points_new[i - 2][1] * frame.shape[0])), clr, 1)
                    frame = cv2.circle(frame, (int(points_new[i][0] * frame.shape[1]), int(points_new[i][1] * frame.shape[0])), 4, clr, -1)

                    INCORRECT += 1
        
        if (time.time() - DEBOUNCE_THRESHOLD) > DEBOUNCE_THRESHOLD_TIME:
            if INCORRECT > MIN_INCORRECT:
                ANGLE_THRESHOLD = min(MAX_THRESHOLD, ANGLE_THRESHOLD + THRESHOLD_MULTIPLIER)
            else:
                ANGLE_THRESHOLD = max(MIN_THRESHOLD, ANGLE_THRESHOLD - THRESHOLD_MULTIPLIER)
            
            DEBOUNCE_THRESHOLD = time.time()
            INCORRECT = 0

        if prev_text:
            cv2.putText(frame, prev_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Next prediction in {DEBOUNCE_TIME - (time.time() - debounce):.2f} sec.', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Dynamic Angle Threshold: {ANGLE_THRESHOLD}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(1) == 27:
            break

        if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()