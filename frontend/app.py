from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import os
import random
from keras.models import load_model

app = Flask(__name__)

# Load the emotion detection model
model = load_model('emotion_model.h5')

# Load face detector (Haar Cascade)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define emotion labels
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Define the music directory
MUSIC_DIR = 'static/music'

# Function to preprocess the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start the webcam
camera = cv2.VideoCapture(0)  # Use 1 for external webcam

# Helper function to predict emotion
def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(frame, 1.3, 5)
    emotion = 'No face detected'  # Default emotion if no faces detected

    if len(faces) == 0:  # No faces detected
        return emotion

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = extract_features(face)
        prediction = model.predict(face)
        emotion = labels[np.argmax(prediction)]

        # Display the emotion on the frame
        cv2.putText(frame, emotion, (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2, cv2.LINE_AA)
    
    return emotion

# Helper function to choose a song based on emotion
def get_song_for_emotion(emotion):
    emotion_path = os.path.join(MUSIC_DIR, emotion)
    if os.path.exists(emotion_path):
        songs = os.listdir(emotion_path)
        if songs:
            return f'{emotion}/{random.choice(songs)}'
    return None

# Video streaming route
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                frame = cv2.flip(frame, 1)
                emotion = predict_emotion(frame)  # Predict emotion from webcam frame
                ret, buffer = cv2.imencode('.jpg', frame)  # Encode frame for streaming
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Route to get emotion and song
@app.route('/get_song', methods=['GET'])
def get_song():
    success, frame = camera.read()
    if success:
        emotion = predict_emotion(frame)  # Predict emotion from webcam frame
        song = get_song_for_emotion(emotion)
        return jsonify({'emotion': emotion, 'song_path': song})
    return jsonify({'error': 'Unable to capture video'}), 500

if __name__ == '__main__':
    app.run(debug=True)
