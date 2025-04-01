import cv2
import imutils
from imutils import face_utils
from scipy.spatial import distance
import dlib
from pygame import mixer

mixer.init()
mixer.music.load(r"C:\Users\hp\Downloads\music.wav")

def eye_aspect_ratio(eye):
    A=distance.euclidean(eye[1],eye[5])
    B=distance.euclidean(eye[2],eye[4])
    C=distance.euclidean(eye[0],eye[3])
    ear=(A+B)/(2.0*C)
    return ear

thresh=0.25
flag=0
frame_check=30

(lStart,lEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS['left_eye']  # eye landmark of left eye
(rStart,rEnd)=face_utils.FACIAL_LANDMARKS_68_IDXS['right_eye']  # eye landmark of right eye

# get_frontal_face_detector() function is a pre-trained face detector. It uses an algorithm called the Haar Cascade Classifier to detect faces in an image or video stream.
# Assigned the result of dlib.get_frontal_face_detector() to the variable detect.
# Now, detect is a face detection object that can be used later to detect faces in an image or frame.

detect=dlib.get_frontal_face_detector()
predict=dlib.shape_predictor(r"C:\Users\hp\Downloads\shape_predictor_68_face_landmarks.dat")       # trained to identify specific points (landmarks) on a face like eyes, eyebrows, nose, mouth, and jawline - herein 68 landmarks


cap=cv2.VideoCapture(0)   # initializes the default camera 0 (webcam) to capture vide
while True:               # start of an infinite loop which executes untill stopped manually
    ret, frame=cap.read() # captures a single frame from the camera and stores it in the variable frame
    frame=imutils.resize(frame,width=600)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    subjects=detect(gray,0) # detected face

    for subject in subjects:                    # subject: This is an individual face from the subjects list.
        shape=predict(gray,subject)            # uses the shape predictor model (initialized earlier as predict using dlib.shape_predictor) to detect the 68 facial landmarks within the subject (the detected face).
        shape=face_utils.shape_to_np(shape)     # shape object (which contains the landmarks) into a NumPy array, so basically shape variable is now a 2D array where each row corresponds to a specific landmark point.
        leftEye=shape[lStart:lEnd]
        rightEye=shape[rStart:rEnd]
        leftEar=eye_aspect_ratio(leftEye)
        rightEar=eye_aspect_ratio(rightEye)
        ear=(leftEar+rightEar)/2.0
        leftEyeHull=cv2.convexHull(leftEye)
        rightEyeHull=cv2.convexHull(rightEye)
        cv2.drawContours(frame,[leftEyeHull],-1,(0,255,0),1)
        cv2.drawContours(frame,[rightEyeHull],-1,(0,255,0),1)
        if ear<thresh:
            flag+=1
            print(flag)
            if flag>=frame_check:
                cv2.putText(frame, "****ALERT****", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                cv2.putText(frame, "****ALERT****", (10,325), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                mixer.music.play()
        else:
            flag=0

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()