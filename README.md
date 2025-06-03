# Drowsiness_Detection_System_Using_OpenCV
# Computer_Vision_Project

This project implements a real-time driver drowsiness detection system using OpenCV, Dlib, and Scipy. It monitors eye aspect ratio (EAR) to detect signs of drowsiness and triggers an alarm if the driver appears sleepy for a sustained period.

🧠 How It Works
The system uses facial landmark detection to locate the eyes and calculates the Eye Aspect Ratio (EAR). When the eyes remain closed for a number of consecutive frames, an alarm is triggered.

1. EAR is computed for both eyes using Euclidean distances between facial landmarks.
2. If EAR drops below a threshold (0.25) for a set number of frames (30), an alert is sounded.
3. The system continuously streams from a webcam and displays the video feed with eye contours and warnings.


