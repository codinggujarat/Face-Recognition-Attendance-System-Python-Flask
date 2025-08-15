from flask import Flask, render_template, Response, request, redirect, url_for, send_file
import cv2
import os
from pathlib import Path
import threading
import time
import pandas as pd
from datetime import datetime
import pickle
import numpy as np

app = Flask(__name__)

# ----------------------------
# Paths & Globals
# ----------------------------
ATTENDANCE_FILE = "attendance.csv"
dataset_dir = Path("dataset")
dataset_dir.mkdir(exist_ok=True)

# Capture faces
capture_name = None
capture_count = 0
capture_target = 50
capture_path = None
capture_in_progress = False

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
camera = cv2.VideoCapture(0)

# Attendance process
attendance_running = False
marked_names = set()
recognizer = None
labels = None

# ----------------------------
# Helper functions
# ----------------------------
def slugify(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")

# ----------------------------
# Capture Faces
# ----------------------------
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
            face_img = cv2.resize(face_img, (200, 200))  # Resize for consistency
            cv2.imwrite(str(capture_path / f"{capture_count}.jpg"), face_img)
        time.sleep(0.05)
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
        if capture_in_progress:
            cv2.putText(frame, f"Capturing: {capture_count}/{capture_target}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if attendance_running and recognizer and labels:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (200,200))  # Resize for prediction
                id_, confidence = recognizer.predict(roi_gray)
                if confidence < 60:
                    name = labels.get(id_, "Unknown")
                    if name not in marked_names:
                        now = datetime.now()
                        with open(ATTENDANCE_FILE, "a") as f:
                            f.write(f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")
                        marked_names.add(name)
                    cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def dashboard():
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name","Date","Time"])
    return render_template("dashboard.html", tables=df.to_dict(orient="records"),
                           capture_in_progress=capture_in_progress,
                           attendance_running=attendance_running)

@app.route("/start_capture", methods=["POST"])
def start_capture():
    global capture_name, capture_count, capture_path, capture_in_progress
    capture_name = request.form.get("name")
    if not capture_name:
        return "Name required!", 400
    name_slug = slugify(capture_name)
    capture_path = dataset_dir / name_slug
    capture_path.mkdir(parents=True, exist_ok=True)
    existing = list(capture_path.glob("*.jpg"))
    capture_count = len(existing)
    capture_in_progress = True
    threading.Thread(target=capture_faces).start()
    return redirect(url_for("dashboard"))

@app.route("/train_model")
def train_model():
    global recognizer, labels
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_samples = []
    ids = []
    label_map = {}
    current_id = 0

    for person_name in os.listdir(dataset_dir):
        person_path = dataset_dir / person_name
        if not person_path.is_dir():
            continue
        label_map[current_id] = person_name
        for img_file in os.listdir(person_path):
            img_path = person_path / img_file
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            faces = detector.detectMultiScale(img)
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))  # Resize for uniform shape
                face_samples.append(face_img)
                ids.append(current_id)
        current_id +=1

    if face_samples:
        recognizer.train(np.array(face_samples), np.array(ids))  # Fixed OpenCV 4+
        recognizer.write("trainer.yml")
        with open("labels.pickle","wb") as f:
            pickle.dump(label_map,f)
    labels = label_map
    return redirect(url_for("dashboard"))

@app.route("/start_attendance")
def start_attendance():
    global attendance_running, recognizer, labels
    if not recognizer or not labels:
        if not os.path.exists("trainer.yml") or not os.path.exists("labels.pickle"):
            return "Train the model first!", 400
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("trainer.yml")
        with open("labels.pickle","rb") as f:
            labels = pickle.load(f)
    attendance_running = True
    return redirect(url_for("dashboard"))

@app.route("/stop_attendance")
def stop_attendance():
    global attendance_running, marked_names
    attendance_running = False
    marked_names = set()
    return redirect(url_for("dashboard"))

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/delete_attendance/<name>", methods=["POST"])
def delete_attendance(name):
    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
        df = df[df["Name"] != name]
        df.to_csv(ATTENDANCE_FILE, index=False)
    return redirect("/")

@app.route("/export_excel")
def export_excel():
    excel_path = "attendance.xlsx"
    pd.read_csv(ATTENDANCE_FILE).to_excel(excel_path, index=False)
    return send_file(excel_path, as_attachment=True)

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        camera.release()
