# 🎯 Face Recognition Attendance System (Python & Flask)

A **web-based face recognition attendance system** built using Python, OpenCV, Flask, and LBPH Face Recognizer. This system allows automated attendance tracking using a webcam, with features like face capture, model training, real-time attendance logging, deletion of records, and exporting attendance data to Excel.

---

## 🛠 Features

- Capture student faces via webcam.
- Train a face recognition model (LBPH).
- Start and stop real-time attendance.
- Display attendance records in a responsive dashboard.
- Delete individual attendance entries.
- Export attendance data to Excel.
- Browser-based webcam feed for live face capture.

---

## 📂 Project Structure

Face_Recognition_Attendance/
│
├─ app.py                 
├─ capture_faces.py        
├─ train_model.py         
├─ attendance.py          
├─ dataset/                
├─ labels.pickle          
├─ attendance.csv         
├─ templates/             
└─ static/                

---

## ⚡ Installation

1. Clone the repository:
```bash
git clone <repo_url>
cd Face_Recognition_Attendance
````

1. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
````
1. Install required packages:
```bash
pip install -r requirements.txt
# (requirements include: Flask, OpenCV, Pillow, pandas, numpy)
````

## 🚀 Usage

Run the Flask app:
python app.py

Open the dashboard:  
Visit `http://127.0.0.1:5000` in your browser.

### Capture Faces
- Enter student name and click **Capture Faces**.
- Webcam opens in the browser and automatically saves 50 face images per student.

### Train Model
- Click **Retrain Model** after capturing faces.
- Generates `trainer.yml` and `labels.pickle` automatically.

### Start Attendance
- Click **Start Attendance** to log attendance in real-time.
- Names are detected automatically and saved in `attendance.csv`.

### Stop Attendance
- Click **Stop Attendance** to end the session.

### Export Excel
- Click **Export to Excel** to download attendance records.

### Delete Attendance
- Use the **Delete** button next to each record to remove it.

---

## 📌 Notes

- Ensure your webcam is connected.
- `labels.pickle` and `trainer.yml` are automatically created when training.
- Face images are stored in `dataset/<student_name>/`.
- Attendance records are stored in `attendance.csv`.

---

## 🖥 Technologies Used

- Python 3.x
- Flask (Web framework)
- OpenCV (Computer vision)
- LBPH Face Recognizer
- Pandas (Data management)
- HTML, Bootstrap (Dashboard)

---

## 🖼 Screenshots

*(Include images of the dashboard, capture window, and attendance table here)*

---

## 🔗 References

- [OpenCV Documentation](https://opencv.org)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Boxicons for icons](https://boxicons.com)

---

## ⚡ Author

**AMAN NAYAK**  
Email: codinggujarat@gmail.com  
GitHub: [codinggujarat](https://github.com/codinggujarat)
