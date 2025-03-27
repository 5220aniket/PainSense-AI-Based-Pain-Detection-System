import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from collections import defaultdict, deque
import os
from flask import Flask, request, jsonify, render_template, Response
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)

# ----------------------------
# 1. Configuration & Constants
# ----------------------------
PAIN_LEVELS = {
    0: {'name': 'No Pain', 'color': (0, 255, 0), 'threshold': 0.7},
    1: {'name': 'Low Pain', 'color': (0, 255, 255), 'threshold': 0.65},
    2: {'name': 'Medium Pain', 'color': (0, 165, 255), 'threshold': 0.6},
    3: {'name': 'High Pain', 'color': (0, 0, 255), 'threshold': 0.55},
    4: {'name': 'Unbearable Pain', 'color': (0, 0, 128), 'threshold': 0.5}
}

SEQ_LENGTH = 45  # 1.5 seconds at 30 FPS
IMG_SIZE = 200  # For image classification model
KEYPOINT_INDICES = {
    'nose': 0,
    'shoulders': [5, 6],
    'hips': [11, 12],
    'knees': [13, 14],
    'elbows': [7, 8]
}

# ----------------------------
# 2. Initialize Models
# ----------------------------
# Pose estimation model
pose_model = YOLO('yolov8n-pose.pt')

# Real-time pain detection model
class PainDetector(tf.keras.Model):
    def __init__(self):
        super(PainDetector, self).__init__()
        self.lstm1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.dropout1 = tf.keras.layers.Dropout(0.3)
        self.lstm2 = tf.keras.layers.LSTM(64)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dropout2 = tf.keras.layers.Dropout(0.4)
        self.output_layer = tf.keras.layers.Dense(5, activation='softmax')
        
    def call(self, inputs):
        x = self.lstm1(inputs)
        x = self.dropout1(x)
        x = self.lstm2(x)
        x = self.dense1(x)
        x = self.dropout2(x)
        return self.output_layer(x)

# Load both models
rt_pain_model = PainDetector()
rt_pain_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image classification model
try:
    img_class_model = load_model("models/pain_recognition_model.h5")
    img_class_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Image classification model loaded successfully!")
except Exception as e:
    print(f"Error loading image model: {str(e)}")

# ----------------------------
# 3. Utility Functions
# ----------------------------
def preprocess_image(img_path):
    """For image classification model"""
    img = Image.open(img_path)
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def generate_frames():
    """Real-time video feed generator"""
    cap = cv2.VideoCapture(0)
    track_history = defaultdict(lambda: {'queue': deque(maxlen=SEQ_LENGTH), 'last_prediction': 0})
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        results = pose_model.track(frame, persist=True, verbose=False)
        annotated_frame = results[0].plot()
        
        if results[0].keypoints is not None:
            for box_id, kps in enumerate(results[0].keypoints.xy.cpu().numpy()):
                track_id = box_id if results[0].boxes.id is None else results[0].boxes.id[box_id].item()
                keypoints = kps.flatten()
                track_history[track_id]['queue'].append(keypoints)
                
                if len(track_history[track_id]['queue']) == SEQ_LENGTH:
                    sequence = np.array(track_history[track_id]['queue'])
                    sequence_norm = (sequence - np.mean(sequence)) / np.std(sequence)
                    prediction = rt_pain_model.predict(np.expand_dims(sequence_norm, 0), verbose=0)[0]
                    
                    track_history[track_id]['last_prediction'] = np.argmax(prediction)
                    confidence = np.max(prediction)
                    pain_info = PAIN_LEVELS[track_history[track_id]['last_prediction']]
                    
                    text = f"ID {track_id}: {pain_info['name']} ({confidence:.2f})"
                    cv2.putText(annotated_frame, text, (10, 30 + (box_id * 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, pain_info['color'], 2)
                    
                    if track_history[track_id]['last_prediction'] >= 3:
                        cv2.putText(annotated_frame, "MEDICAL ATTENTION NEEDED!", 
                                   (frame.shape[1]-400, 30 + (box_id * 30)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

# ----------------------------
# 4. Flask Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        os.makedirs('uploads', exist_ok=True)
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        try:
            img_array = preprocess_image(file_path)
            prediction = img_class_model.predict(img_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            pain_level = ['No pain', 'Low pain', 'Medium pain', 'High pain', 'Unbearable pain'][predicted_class]
            
            os.remove(file_path)
            return jsonify({
                'predicted_pain_level': pain_level,
                'confidence': float(np.max(prediction))
            })
        except Exception as e:
            return jsonify({'error': str(e)})

# ----------------------------
# 5. Run the Application
# ----------------------------
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)