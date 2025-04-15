from flask import Flask, render_template, Response
import cv2
import torch
from camera import ResNet9, predict_image, class_names
import numpy as np

app = Flask(__name__)

# Global variables
model = None
device = None

def init_model():
    global model, device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet9(3, len(class_names))
    model.load_state_dict(torch.load('plant-disease-model.pth', map_location=device))
    model.to(device)
    model.eval()

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([30, 40, 40])
        upper_yellow = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_yellow)
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        rgb_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        if np.count_nonzero(mask) > 1000:
            prediction = predict_image(rgb_frame, model)
        else:
            prediction = "No plant detected"
            
        # Add prediction text
        cv2.putText(frame, prediction, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Convert frame to bytes
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    init_model()
    app.run(debug=True)