import cv2
import atexit
from flask import Flask, render_template, Response

app = Flask(__name__)
cap = None  # Initialize the camera outside

def init_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

def generate_frames():
    global cap  # Use the global camera object
    init_camera()  # Ensure the camera is initialized
    while True:
        # Capture frame-by-frame
        success, frame = cap.read()
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in the proper format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@atexit.register
def cleanup():
    if cap is not None:
        cap.release()

if __name__ == "__main__":
    # Disable reloader in debug mode to avoid double initialization
    app.run(debug=True, use_reloader=False)
