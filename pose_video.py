import cv2
import copy
import time
import itertools
import csv
import ast
import torch
import joblib

import mediapipe as mp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from tensorflow.keras.models import load_model # type: ignore
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


def calculate_accuracy(accuracy_history, weight=0.1):
    accuracy = [((accuracy_history[i][0] - accuracy_history[i][1] - (accuracy_history[i][2] * weight)) / accuracy_history[i][0]) * 100 for i in range(len(accuracy_history))]
    return accuracy


def create_pie_chart(accuracy_history):
    plt.tight_layout()
    accuracy_history = np.array(accuracy_history)

    red_sum = np.sum(accuracy_history[:, 1])
    yellow_sum = np.sum(accuracy_history[:, 2])
    green_sum = np.sum(accuracy_history[:, 0]) - red_sum - yellow_sum

    sums = [red_sum, yellow_sum, green_sum]
    colors = ['r', 'y', 'g']
    labels = ['Flawed', 'Imperfect', 'Accurate']

    filtered_data = [(s, c, l) for s, c, l in zip(sums, colors, labels) if s > 0]
    sums, colors, labels = zip(*filtered_data) if filtered_data else ([], [], [])

    plt.title('Pose Distribution')
    plt.pie(sums, colors=colors, labels=labels)
    fig.canvas.draw()

    pie = np.frombuffer(fig.canvas.get_renderer().buffer_rgba(), dtype=np.uint8)

    width, height = fig.canvas.get_width_height()
    pie = pie.reshape((height, width, 4))
    pie = cv2.cvtColor(pie, cv2.COLOR_RGBA2BGR)

    fig.clf()

    return pie


def create_line_plot(accuracy):
    plt.tight_layout()
    plt.plot(accuracy)
    plt.title('Accuracy')

    plt.xlabel('Time')
    plt.ylabel('Accuracy (%)')

    plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

    ax = plt.gca()
    ax.set_xticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.canvas.draw()

    plot = np.frombuffer(fig.canvas.get_renderer().buffer_rgba(), dtype=np.uint8)

    width, height = fig.canvas.get_width_height()
    plot = plot.reshape((height, width, 4))
    plot = cv2.cvtColor(plot, cv2.COLOR_RGBA2BGR)

    fig.clf()

    return plot


def landmark_in_frame(landmark):
    return 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1 and landmark.visibility > 0.4


def full_body_in_frame(landmarks):
    required_indices = [
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE,
        mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
    ]
    
    return all(
        landmark_in_frame(landmarks[i]) for i in required_indices
    )


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]    # Take output of last time step
        out = self.fc(out)
        out = torch.softmax(out, dim=1)
        return out


class Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.device = None
        self.type = None
        self.model = None


    def set_type(self):
        match self.model_path:
            case str() if self.model_path.endswith('.keras'):
                self.type = 'tf'
            case str() if self.model_path.endswith('.pth'):
                self.type = 'torch'
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            case str() if self.model_path.endswith('.pkl'):
                self.type = 'sklearn'
            case _:
                raise ValueError("Unsupported model file type.")
    

    def load_model(self):
        if self.set_type is None:
            raise ValueError("Model type not set. Please call set_type() first.")
        
        match self.type:
            case 'tf':
                self.model = load_model(self.model_path)
            case 'torch':
                if self.device is None:
                    raise ValueError("Device not set. Please call set_type() first.")
                
                self.model = LSTMClassifier(input_size=36, hidden_size=16, num_classes=8, num_layers=1).to(self.device)
                self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
                self.model.eval()
            case 'sklearn':
                self.model = joblib.load(self.model_path)
    

    def predict(self, data):
        if self.type is None:
            raise ValueError("Model type not set. Please call set_type() first.")

        match self.type:
            case 'tf':
                predict_model = self.model.predict(data, verbose=0)
                return np.squeeze(predict_model), np.argmax(predict_model, axis=1)[0]
            case 'torch':
                tensor = data.reshape(-1, 1, 36)
                tensor = torch.tensor(tensor, dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    predict_model = self.model(tensor)
                    return np.squeeze(np.array(predict_model)), torch.argmax(predict_model, dim=1).item()
            case 'sklearn':
                predict_model = self.model.predict_proba(data)
                return np.squeeze(predict_model), np.argmax(self.model.predict_proba(data), axis=1)[0]


if __name__ == "__main__":
    correction_folder = Path(__file__).resolve().parent / 'Correction_Data'
    video_folder = Path(__file__).resolve().parent / 'Tests/Video'
    audio_folder = Path(__file__).resolve().parent / 'Audio'

    v1 = str(video_folder / 'Hasta Uttanasan/1_HU.mp4')
    v2 = str(video_folder / 'Panchim Uttanasan/1_PU.mp4')
    v3 = str(video_folder / 'Vrikshasana/1_V.mp4')
    v4 = str(video_folder / 'Vajrasana/1.mp4')
    v5 = str(video_folder / 'Tadasana/1.mp4')
    v6 = str(video_folder / 'Padmasana/1.mp4')

    t3_1 = str(video_folder / 'Vrikshasana/Test1.mp4')
    t3_2 = str(video_folder / 'Vrikshasana/Test2.mp4')

    t7 = str(video_folder / 'Bhujangasana/1_B.mp4')

    model_name = 'DNN_Model.keras'
    model_data = str(Path(__file__).resolve().parent / f'Model/FinalModels/{model_name}')
    model = Model(model_data)
    model.set_type()
    model.load_model()

    cap = cv2.VideoCapture(t3_1)

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(enable_segmentation=False, model_complexity=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

    min_x, max_x, min_y, max_y = 0, 0, 0, 0
    PADDING_CROP = 100
    AUTOMATIC_CROP = False

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

    fig = plt.figure()
    PLOT_COUNT_THRESHOLD = 50
    WARNING_ACCURACY_THRESHOLD = 0.25
    PLOT_SIZE = 150

    _predict = None
    PREDICT_THRESHOLD = 0.8
    predictions = ['Hasta Uttanasan', 'Panchim Uttanasan', 'Vrikshasana', 'Vajrasana', 'Taadasana', 'Padmasana', 'Bhujangasana']
    DEBOUNCE_TIME = 1
    debounce = 0
    prev_text = None
    prev_accuracy = None
    perform_detect = True

    SHOW_GRAPH = False
    accuracy_history = []
    accuracy = []

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

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        if not ret:
            print("Can't receive frame (Video end?). Exiting ...")
            break

        height, width, _ = frame.shape
        FRAME_COUNT += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if (time.time() - debounce) > DEBOUNCE_TIME:
            debounce = time.time()
            perform_detect = True

        if results.pose_landmarks:
            if full_body_in_frame(results.pose_landmarks.landmark):
                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    if mp_pose.PoseLandmark(idx).name in poses:
                        index_pose = poses.index(mp_pose.PoseLandmark(idx).name)
                        points_new_coll[index_pose] = np.array((landmark.x, landmark.y))
                points_new_coll[-1] = (np.array(((points_new_coll[9][0] + points_new_coll[10][0]) / 2, (points_new_coll[9][1] + points_new_coll[10][1]) / 2)))

                points_new_coll = np.array(points_new_coll)
                if AUTOMATIC_CROP:
                    min_x = round(np.min(points_new_coll[:, 0]) * width, -2)
                    max_x = round(np.max(points_new_coll[:, 0]) * width, -2)
                    min_y = round(np.min(points_new_coll[:, 1]) * height, -2)
                    max_y = round(np.max(points_new_coll[:, 1]) * height, -2)

                    if abs(min_x - max_x) > 50 and abs(min_y - max_y) > 50:
                        min_x = max(min_x - PADDING_CROP, 0)
                        max_x = min(max_x + PADDING_CROP, width)
                        min_y = max(min_y - PADDING_CROP, 0)
                        max_y = min(max_y + PADDING_CROP, height)

                if perform_detect:
                    perform_detect = False

                    norm = landmark_list(frame, points_new_coll)
                    predict_model, prediction = model.predict(np.array([norm]))
                    
                    if prediction != 7: # NoAsana
                        if predict_model[prediction] >= PREDICT_THRESHOLD:
                            _predict = prediction

                        if _predict is not None:
                            prev_text = predictions[_predict]
                            prev_accuracy = predict_model[prediction]
                    else:
                        prev_text = 'No Asana'
                        prev_accuracy = predict_model[prediction]
            else:
                prev_text = 'No Asana: Partial Body'
                prev_accuracy = None

            if prev_text and prev_text != 'No Asana' and prev_text != 'No Asana: Partial Body':
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

                accuracy_history.append([len(points_new), 0, 0])
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
                            accuracy_history[-1][1] += 1
                        else:
                            clr = (0, 255, 255) # Yellow
                            accuracy_history[-1][2] += 1

                        frame = cv2.circle(frame, (int(points_new[i - 2][0] * width), int(points_new[i - 2][1] * height)), 4, clr, -1)
                        frame = cv2.circle(frame, (int(points_new[i + av][0] * width), int(points_new[i + av][1] * height)), 4, clr, -1)

                        frame = cv2.line(frame, (int(points_new[i][0] * width), int(points_new[i][1] * height)), (int(points_new[i + av][0] * width), int(points_new[i + av][1] * height)), clr, 1)

                    if show_all_points:
                        frame = cv2.line(frame, (int(points_new[i][0] * width), int(points_new[i][1] * height)), (int(points_new[i - 2][0] * width), int(points_new[i - 2][1] * height)), clr, 1)
                        frame = cv2.circle(frame, (int(points_new[i][0] * width), int(points_new[i][1] * height)), 4, clr, -1)

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

        if AUTOMATIC_CROP:
            if abs(min_x - max_x) > 0 and abs(min_y - max_y) > 0:
                cropped_frame = frame[int(min_y):int(max_y), int(min_x):int(max_x)]

                orig_h, orig_w = cropped_frame.shape[:2]
                target_w, target_h = width, height

                scale = min(target_w / orig_w, target_h / orig_h)

                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)

                text_w = 0
                if prev_text:
                    (text_w, _), _ = cv2.getTextSize(prev_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

                padded_size = max(text_w, 200)

                final_w = max(target_w, new_w + padded_size)

                resized_frame = cv2.resize(cropped_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                padded_frame = np.ones((target_h, final_w, 3), dtype=np.uint8) * 255

                x_offset = padded_size
                y_offset = (target_h - new_h) // 2

                padded_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
                frame = padded_frame

        if accuracy_history and SHOW_GRAPH:
            accuracy_history = accuracy_history[-PLOT_COUNT_THRESHOLD:]
            accuracy = calculate_accuracy(accuracy_history, weight=WARNING_ACCURACY_THRESHOLD)

            pie = create_pie_chart(accuracy_history)
            plot = create_line_plot(accuracy)

            pie_height, pie_width, _ = pie.shape
            plot_height, plot_width, _ = plot.shape

            new_plot_height = int(plot_height * (PLOT_SIZE / plot_width))
            plot_resized = cv2.resize(plot, (PLOT_SIZE, new_plot_height), interpolation=cv2.INTER_AREA)

            new_pie_height = int(pie_height * (PLOT_SIZE / pie_width))
            pie_resized = cv2.resize(pie, (PLOT_SIZE, new_pie_height), interpolation=cv2.INTER_AREA)

            padding = 10
            plot_x, plot_y = padding, frame.shape[0] - new_plot_height - new_pie_height - 2 * padding
            pie_x, pie_y = padding, plot_y + new_plot_height + padding

            frame[plot_y:plot_y+new_plot_height, plot_x:plot_x+PLOT_SIZE] = plot_resized
            frame[pie_y:pie_y+new_pie_height, pie_x:pie_x+PLOT_SIZE] = pie_resized

        if prev_text:
            after_prev = f' ({round(prev_accuracy * 100, 2)}%)' if prev_accuracy else ''
            cv2.putText(frame, f'{prev_text}{after_prev}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        if accuracy:
            cv2.putText(frame, f'Asana Avg. Accuracy: {round(sum(accuracy) / len(accuracy), 2)}%', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Video', frame)

        end = cv2.waitKey(1)
        if end & 0xFF == ord('q'):
            break
        elif end & 0xFF == ord('p'):
            show_all_points = not show_all_points
        elif end == 27:
            break
        elif end & 0xFF == ord('z'):
            AUTOMATIC_CROP = not AUTOMATIC_CROP
        elif end & 0xFF == ord('g'):
            SHOW_GRAPH = not SHOW_GRAPH

        if cv2.getWindowProperty('Video', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()