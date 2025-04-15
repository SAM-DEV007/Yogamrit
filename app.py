from flask import Flask, render_template, Response, jsonify
import cv2

from core_logic_reframed import process_frame, calculate_angle, preprocess_data, landmark_list

app = Flask(__name__)

# Global variables
current__asana = None



def generate_frames():
    global current__asana
    
    cap = cv2.VideoCapture(0)
    
    # Set camera properties to ensure original frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Set desired width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)  # Set desired height
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    try:
        while True:
            frame, prev_text = process_frame(cap)
            current__asana = prev_text
            

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    except Exception as e:
        print(f"Error in generate_frames: {e}")
    finally:
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
    global current__asana
    return jsonify({"asana": current__asana})



if __name__ == '__main__':
    app.run(debug=True)