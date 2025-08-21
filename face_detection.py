import cv2
import numpy as np
from deepface import DeepFace

# Load Haar cascade classifiers for face, eyes, and smile
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

video_cap = cv2.VideoCapture(0)  # Capture video from the default camera

if not video_cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

while True:
    ret, video_data = video_cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    try:
        # Use DeepFace to analyze emotions
        analysis = DeepFace.analyze(video_data, actions=['emotion'], enforce_detection=False)
        emotion = analysis[0]['dominant_emotion']
        # actions=['emotion']: It tells DeepFace to analyze only emotions (ignoring age, gender, etc.).
        # enforce_detection=False: Allows processing even if a face is not detected properly


        # Display emotion on screen
        cv2.putText(video_data, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print(f"DeepFace Error: {e}")


    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = video_data[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)  # Yellow for smiles

    cv2.imshow("Face, Eye & Smile Detection + Emotion Recognition", video_data)

    if cv2.waitKey(10) == ord("s"):
        cv2.imwrite("smile_captured.jpg", video_data)
        print("Image saved as 'smile_captured.jpg'")

    elif cv2.waitKey(10) == ord("v"):
        break

video_cap.release()
cv2.destroyAllWindows()
