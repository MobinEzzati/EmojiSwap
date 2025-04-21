# 🎭 Emoji Swap – Real-Time Face-to-Emoji App

**Emoji Swap** is a real-time face filter app that detects human facial expressions and replaces the face with the closest matching emoji. Using computer vision and facial landmark detection, the app tracks facial movements and overlays expressive emojis that mirror your mood — live on your webcam!

---

## 🚀 Features

- 🔍 **Real-time face detection** using dlib or MediaPipe
- 🧠 **Facial expression analysis**: surprised, sad, angry, neutral, etc.
- 😀 **Emoji overlay**: Matches your expression to the perfect emoji
- 🖼️ **Transparent emoji blending** for realistic overlays
- 📸 **Supports multiple expressions** and easily extendable emoji sets
- 💻 Lightweight, fun, and runs locally — no need for cloud services

---

## Author
This app is made by Mobin Ezzati, SMU computer science grad student. I am constantly looking forward to improving myself and my knowledge to become a better engineer and make this world more pleasent by technology. 



## 🛠️ Tech Stack

- `Python 3`
- `OpenCV` – webcam access and image processing
- `dlib` – facial landmark detection (MediaPipe alternative available)
- `NumPy` – efficient image manipulation
- `imutils` – helper functions for image transformation

---

## 💡 How It Works

1. App captures frames from your webcam
2. Facial landmarks are detected in real-time
3. Simple logic or ML models classify your facial expression
4. The matching emoji is resized and overlaid on your face area
5. Magic happens — your face becomes a live emoji!

---

## 🌈 Future Upgrades

- MediaPipe integration for faster multi-face support
- Deep learning model for advanced expression recognition
- Face tracking & emoji animation
- Export as GIF/video or integrate with streaming tools

---

## 📸 Demo

![emoji swap demo gif](#)  
<!-- Add your screen recording or demo gif here -->

---

## 📁 Folder Structure
