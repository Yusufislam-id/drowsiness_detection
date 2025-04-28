import cv2
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer


is_drowsy = False 
# is_drowsy3 = False
numbers = 0
drowsy_duration = 6
mixer.init()
alarm_sound = mixer.Sound('short-success-sound.wav')

model1 = load_model('models/face-detection2.h5', compile=False)


def preprocess_image(face_image):
    processed_image = cv2.resize(face_image, (224, 224))  # Ubah ukuran gambar
    processed_image = processed_image / 255.0  # Normalisasi
    return processed_image

def detect_eye(face_roi):
    processed_image = preprocess_image(face_roi)  
    prediction = model1.predict(np.expand_dims(processed_image, axis=0))
    return prediction[0][0] < 0.5

face_detection = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
# left_eye_detection = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
# right_eye_detection = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

# Mulai akses webcam
cap = cv2.VideoCapture(0)  # Gunakan 0 jika webcam default

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))

    for (x,y,w,h) in faces:
        face_roi = frame[y:y + h, x:x + w]
        is_drowsy = detect_eye(face_roi)
        if (numbers >= drowsy_duration):
            cv2.putText(frame, '!!!Alert, Wake Up', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 31, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # bunyikan alarm
            try:
                alarm_sound.play()
            except:  # isplaying = False
                pass
        else:
            cv2.putText(frame, 'Awake', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (85, 225, 80), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (85, 225, 80), 2)
            mixer.stop()
    
    if (is_drowsy):
        numbers = numbers + 1
    else:
        numbers = 0
    
    cv2.imshow('Drowsiness Detection', frame)
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()