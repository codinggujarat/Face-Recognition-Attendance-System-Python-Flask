from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import os
from pathlib import Path
import threading
import time

app = Flask(__name__)

dataset_dir = Path("dataset")
dataset_dir.mkdir(exist_ok=True)

# Global variables to manage capture
capture_name = None
capture_count = 0
capture_target = 50
capture_path = None
capture_in_progress = False

# Initialize face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
camera = cv2.VideoCapture(0)

def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")

def capture_faces():
    global capture_count, capture_in_progress
    while capture_in_progress and capture_count < capture_target:
        ret, frame = camera.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            capture_count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(str(capture_path / f"{capture_count}.jpg"), face_img)
        time.sleep(0.05)  # small delay to control speed
    capture_in_progress = False

def gen_frames():
    global capture_count, capture_in_progress
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Overlay progress
        if capture_in_progress:
            cv2.putText(frame, f"Capturing: {capture_count}/{capture_target}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('capture_browser.html', capture_in_progress=capture_in_progress)

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capture_name, capture_count, capture_path, capture_in_progress
    capture_name = request.form.get("name")
    if not capture_name:
        return "Name required!", 400
    name_slug = slugify(capture_name)
    capture_path = dataset_dir / name_slug
    capture_path.mkdir(parents=True, exist_ok=True)
    # Count existing images
    existing = list(capture_path.glob("*.jpg"))
    capture_count = len(existing)
    capture_in_progress = True
    # Start capture in a separate thread
    threading.Thread(target=capture_faces).start()
    return redirect(url_for('index'))

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/capture_faces_browser")
def capture_faces_browser():
    return redirect("/capture_faces_flask")  # the route of your browser-based capture app

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        camera.release()
