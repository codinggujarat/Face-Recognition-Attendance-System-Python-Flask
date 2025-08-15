import cv2
import numpy as np
import os
import pickle

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(dataset_path):
    face_samples = []
    ids = []
    label_map = {}
    current_id = 0

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        label_map[current_id] = person_name
        for img_file in os.listdir(person_path):
            img_path = os.path.join(person_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces = detector.detectMultiScale(img)
            for (x, y, w, h) in faces:
                face_img = img[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (200, 200))  # Resize to fixed shape
                face_samples.append(face_img)
                ids.append(current_id)
        current_id += 1

    return face_samples, ids, label_map

# Prepare training data
faces, ids, label_map = get_images_and_labels("dataset")

if faces and ids:
    # Train recognizer
    recognizer.train(np.array(faces), np.array(ids))
    recognizer.write("trainer.yml")  # save model

    # Save label mapping
    with open("labels.pickle", "wb") as f:
        pickle.dump(label_map, f)

    print("Training complete. trainer.yml and labels.pickle created successfully.")
else:
    print("No faces found in dataset. Please check dataset folders.")
