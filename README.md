

Facial Emotion Detection using CNN – Detailed Project Description

This project focuses on real-time facial emotion recognition using a Convolutional Neural Network (CNN). The goal is to detect human emotions such as happy, sad, angry, surprised, etc., from facial expressions using a webcam feed. The system first captures the video in real time using OpenCV and then detects the face from each frame using a Haar Cascade classifier. Once a face is detected, the face region is preprocessed and passed to a trained CNN model that classifies the emotion.

The dataset used for training the model is the FER-2013 (Facial Expression Recognition 2013) dataset, which is publicly available on Kaggle. This dataset contains grayscale images of size 48x48 pixels, each labeled with one of seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. These images are of real human faces with varying facial expressions and were collected for the purpose of emotion detection tasks. Each image is already preprocessed to be centered on the face, which makes it suitable for CNN training.

To train the model, a Convolutional Neural Network was built using TensorFlow and Keras. The model includes multiple convolutional layers for feature extraction followed by dense layers for classification. After training, the model was saved in two parts: the architecture was saved in model.json and the weights were saved in model-2.h5. This separation allows easy reuse of the model without the need to retrain every time.

For real-time emotion detection, a Python script (main.py) was developed. This script captures live video using OpenCV, uses Haar Cascade (haarcascade_frontalface_default.xml) to detect faces in the video frames, and extracts the region of interest (ROI) where the face is located. The ROI is resized to 48x48 pixels to match the input shape expected by the CNN model. The pixel values are normalized, and the image is reshaped before being passed to the model for emotion prediction. The predicted emotion label is then displayed on the video feed above the detected face.

To run the project, a few Python packages need to be installed including opencv-python, tensorflow, keras, and numpy. Once the requirements are installed, the user can simply run the main.py script, and the system will start detecting emotions in real time using the webcam. This makes the project suitable for live demonstrations and practical applications like emotion-aware apps or human-computer interaction systems.

In summary, this project showcases a complete pipeline from data preprocessing and model training to real-time implementation of facial emotion recognition. It not only demonstrates the power of CNNs in understanding visual patterns but also shows how these models can be applied to solve real-world problems. Future improvements can include adding support for multiple faces, improving accuracy by using deeper models or transfer learning, and creating a user-friendly GUI to make the application more interactive.




