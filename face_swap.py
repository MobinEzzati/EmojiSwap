import cv2
import dlib
import numpy as nppip
from pip._internal.cli.cmdoptions import python

import os

import cv2
import dlib
import numpy as np

print("✅ OpenCV version:", cv2.__version__)
print("✅ dlib version:", dlib.__version__)
print("✅ All set!")



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load emoji images
def load_emojis(path="emojis"):
    emojis = {}
    for file in os.listdir(path):
        if file.endswith(".png"):
            name = os.path.splitext(file)[0]
            img = cv2.imread(os.path.join(path, file), cv2.IMREAD_UNCHANGED)  # Keep alpha
            emojis[name] = img
    return emojis

emoji_dict = load_emojis()

def mouth_open(landmarks):
    top_lip = landmarks.part(62).y
    bottom_lip = landmarks.part(66).y
    return bottom_lip - top_lip > 15

def eyebrows_raised(landmarks):
    eye_y = landmarks.part(36).y  # Left eye
    brow_y = landmarks.part(19).y  # Left eyebrow
    return eye_y - brow_y > 15

def is_sad(landmarks):
    # Inner eyebrows (21 and 22) vs eyes (37 and 44)
    brow_avg = (landmarks.part(21).y + landmarks.part(22).y) / 2
    eye_avg = (landmarks.part(37).y + landmarks.part(44).y) / 2
    brows_up = eye_avg - brow_avg > 8

    # Mouth corners (48 and 54) vs center lip (66)
    left_corner = landmarks.part(48).y
    right_corner = landmarks.part(54).y
    center_lip = landmarks.part(66).y
    mouth_down = center_lip - ((left_corner + right_corner) / 2) < -5

    return brows_up and mouth_down

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # Detect expression
        expression = "cool"
        if is_sad(landmarks):
            expression = "sad"

        if mouth_open(landmarks) and eyebrows_raised(landmarks):
            expression = "shocked"
        elif not mouth_open(landmarks) and not eyebrows_raised(landmarks):
            expression = "sad"

        # Draw rectangle on face
        x1, y1 = face.left(), face.top()
        x2, y2 = face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Load corresponding emoji
        emoji = emoji_dict.get(expression)
        if emoji is not None:
            # Resize emoji to face size
            emoji_resized = cv2.resize(emoji, (x2 - x1, y2 - y1))

            # Split emoji channels
            if emoji_resized.shape[2] == 4:
                emoji_rgb = emoji_resized[:, :, :3]
                alpha = emoji_resized[:, :, 3] / 255.0

                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (
                        frame[y1:y2, x1:x2, c] * (1 - alpha) +
                        emoji_rgb[:, :, c] * alpha
                    )

    cv2.imshow("Emoji Face Swap", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# Load face detector and shape predictor
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
# def load_emojis(path="Emojis"):
#     emojis = {}
#     for file in os.listdir(path):
#         if file.endswith(".png"):
#             name = os.path.splitext(file)[0]
#             img = cv2.imread(os.path.join(path, file), cv2.IMREAD_UNCHANGED)  # Keep alpha
#             emojis[name] = img
#     return emojis
#
# emoji_dict = load_emojis()
#
# print(emoji_dict)

