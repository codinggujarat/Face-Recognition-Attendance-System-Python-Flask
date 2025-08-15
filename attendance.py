import cv2
import pickle
import os
from datetime import datetime
import signal
import sys

# Paths for trained model and labels
trainer_path = "trainer.yml"
labels_path = "labels.pickle"

# Exit gracefully on signals
def signal_handler(sig, frame):
    print("[INFO] Exiting attendance...")
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)   # CTRL+C
signal.signal(signal.SIGTERM, signal_handler)  # Terminate

# Check model and labels
if not os.path.exists(trainer_path) or not os.path.exists(labels_path):
    print("[ERROR] trainer.yml or labels.pickle not found. Run train_model.py first.")
    exit()

# Load trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

# Load labels (ID -> Name)
with open(labels_path, "rb") as f:
    labels = pickle.load(f)  # {id: name}

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Attendance file
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Date,Time\n")

marked_names = set()

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Starting webcam. Press 'q' to quit.")

cv2.namedWindow("Attendance System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Attendance System", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id_, confidence = recognizer.predict(roi_gray)

        if confidence < 60:
            name = labels.get(id_, "Unknown")
            color = (0, 255, 0)
            if name not in marked_names:
                now = datetime.now()
                with open(attendance_file, "a") as f:
                    f.write(f"{name},{now.strftime('%Y-%m-%d')},{now.strftime('%H:%M:%S')}\n")
                marked_names.add(name)
                print(f"[INFO] Marked attendance for {name}")
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Attendance session ended.")
