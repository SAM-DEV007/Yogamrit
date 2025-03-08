from flask import Flask, render_template, Response,jsonify
import cv2
from core_logic_reframed import process_frame, calculate_angle,preprocess_data,landmark_list  # Import core logic

app = Flask(__name__)



def generate_frames():
    cap = cv2.VideoCapture(0)  # Open webcam
    global current__asana

    while True:
        

        frame,prev_text = process_frame(cap)  # Pass frame to process_frame()
        current__asana = prev_text

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('test.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/asana_name')
def asana_name():
    return jsonify({"asana": current__asana})
    

if __name__ == '__main__':
    app.run(debug=True)
