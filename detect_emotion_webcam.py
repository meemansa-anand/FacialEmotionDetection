import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# Dynamically load emotion labels from folder names
emotion_labels = sorted(os.listdir('images/train'))  # Load folder names as emotion labels
print("Emotion labels:", emotion_labels)  # Sanity check

# Load trained model
model = load_model('model.h5')

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray_frame[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray / 255.0
        roi = roi.reshape(1, 48, 48, 1)

        prediction = model.predict(roi)
        max_index = np.argmax(prediction[0])
        
        # Debugging: print prediction and max index
        print("Prediction:", prediction)
        print("Max index:", max_index)

        emotion = emotion_labels[max_index]

        # Make the rectangle around the face thicker (thickness = 4)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)

        # Make the text bold/thicker (thickness = 3)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('Facial Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


