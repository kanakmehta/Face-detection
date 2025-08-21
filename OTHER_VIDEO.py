import cv2
import numpy as np
from fer import FER  # Facial Expression Recognition
import time
from pytube import YouTube
import os

# Function to download a YouTube video
def download_video(youtube_url, save_path="video.mp4"):
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(file_extension="mp4", progressive=True).first()
    stream.download(filename=save_path)
    return save_path

# Provide the YouTube video URL
video_url = "https://www.youtube.com/watch?v=your_video_id"  # Replace with your video URL
video_path = download_video(video_url)

# Load Haar cascade classifiers for face, eyes, and smile detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

emotion_detector = FER()

# Load emoji images
emoji_images = {
    "happy": cv2.imread("happy_emoji.png", cv2.IMREAD_UNCHANGED),  
    "sad": cv2.imread("sad_emoji.png", cv2.IMREAD_UNCHANGED),
    "angry": cv2.imread("angry_emoji.png", cv2.IMREAD_UNCHANGED),
}

# Open the downloaded video file instead of a webcam
video_cap = cv2.VideoCapture(video_path)

if not video_cap.isOpened():
    print("Error: Could not open video file.")
    exit()

def overlay_emoji(frame, emoji, x, y, w, h):
    """ Overlay an emoji on the detected face. """
    emoji = cv2.resize(emoji, (w, h))  # Resize emoji to fit the face
    for i in range(h):
        for j in range(w):
            if emoji[i, j][3] != 0:  # Check transparency
                frame[y + i, x + j] = emoji[i, j][:3]
    return frame

while True:
    ret, video_data = video_cap.read()
    if not ret:
        break  # Stop when video ends

    gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)

    # Detect emotions using FER
    emotion, score = emotion_detector.top_emotion(video_data)

    if emotion:
        cv2.rectangle(video_data, (40, 30), (250, 70), (0, 0, 0), -1)
        cv2.putText(video_data, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(video_data, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = video_data[y:y + h, x:x + w]

        # Apply emoji filter if emotion detected
        if emotion in emoji_images and emoji_images[emotion] is not None:
            video_data = overlay_emoji(video_data, emoji_images[emotion], x, y, w, h)

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.2, 5)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detect smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 255), 2)  # Yellow for smiles

    cv2.imshow("Emotion-Based Filters", video_data)

    # Save image with a countdown if 's' is pressed
    if cv2.waitKey(10) == ord("s"):
        print("Capturing image in 3 seconds...")
        time.sleep(3)
        cv2.imwrite("captured_with_filter.jpg", video_data)
        print("Image saved as 'captured_with_filter.jpg'")

    # Exit if 'v' is pressed
    elif cv2.waitKey(10) == ord("v"):
        break

video_cap.release()
cv2.destroyAllWindows()

# Remove the downloaded video file to free up space
os.remove(video_path)
