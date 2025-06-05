# Age and Gender Detection 👤📸

A computer vision project using OpenCV and NumPy to detect a person’s **age range** and **gender** from a webcam or image input. This project uses pre-trained deep learning models for face detection and classification.

---

## 🧠 Tech Stack

- 🐍 Python 3
- 📦 OpenCV (`cv2`)
- 🔢 NumPy
- 🧠 Pre-trained Caffe models (`.caffemodel` & `.prototxt`)

---

## 🎯 Features

- Real-time webcam-based face detection
- Age range prediction (e.g., `0-2`, `4-6`, `25-32`)
- Gender prediction (`Male` / `Female`)
- Works with webcam or static image input

---

## 🚀 How to Run

### 🔧 1. Install Dependencies
```bash
pip install numpy opencv-python
📂 Project Structure
age-and-gender-detection/
├── detect.py
├── age_net.caffemodel
├── gender_net.caffemodel
├── deploy_age.prototxt
├── deploy_gender.prototxt
├── requirements.txt
└── README.md 
🙋‍♂️ Author
Seshathri
🔗 GitHub Profile
