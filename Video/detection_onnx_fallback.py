import cv2 as cv
import time
from datetime import datetime
import numpy as np
import os
import pywhatkit as pwk
import torch
import torch.nn as nn
import torch.nn.functional as F

print("🚀 Starting HAR Detection Script (ONNX with PyTorch fallback)...")
print(f"📁 Current directory: {os.getcwd()}")
print(f"🔍 Looking for video file: test_videos/walk.mp4")

# Define 3D CNN model architecture (matching your training)
class CNN3D(nn.Module):
    def __init__(self, num_classes):
        super(CNN3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3,3,3), padding=1)
        self.pool = nn.MaxPool3d((1,2,2))
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3,3,3), padding=1)
        self.fc1 = nn.Linear(64 * 16 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x is [batch, 16, 3, 112, 112] from training
        x = x.permute(0, 2, 1, 3, 4)  # [batch, 3, 16, 112, 112]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Load actions first
try:
    with open('actions.txt', 'r') as f:
        actions = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(actions)} actions from actions.txt")
except FileNotFoundError:
    print("❌ actions.txt not found!")
    actions = ["unknown"]
except Exception as e:
    print(f"❌ Error loading actions.txt: {e}")
    actions = ["unknown"]

# Try to load ONNX model, fallback to PyTorch
def load_model():
    # Try ONNX first
    try:
        import onnxruntime as ort
        session = ort.InferenceSession("model/HAR.onnx", providers=['CPUExecutionProvider'])
        print("✅ ONNX Model loaded successfully")
        return ("onnx", session)
    except ImportError:
        print("❌ ONNX Runtime not available")
        print("🔄 Falling back to PyTorch model...")
    except Exception as e:
        print(f"❌ ONNX Model loading failed: {e}")
        print("🔄 Falling back to PyTorch model...")
        
    # Fallback to PyTorch
    try:
        model = CNN3D(num_classes=len(actions))
        checkpoint = torch.load("model/cnn3d_hmdb51.pth", map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        print("✅ PyTorch Model loaded successfully")
        return ("pytorch", model)
    except Exception as e2:
        print(f"❌ PyTorch Model loading failed: {e2}")
        return (None, None)

model_type, model = load_model()

# Video setup
VIDEO_SOURCE = "test_videos/walk.mp4"
cap = cv.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("Video file not found, trying webcam...")
    cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("❌ No video source available")
    exit()

# Constants for 3D CNN
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112  # Match training

# Preprocessing for both models
def preprocess_frames(frames, model_type):
    """Preprocess 16 frames for model input"""
    try:
        processed_frames = []
        for frame in frames:
            # Convert BGR to RGB
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            # Resize to SAMPLE_SIZE
            frame_resized = cv.resize(frame_rgb, (SAMPLE_SIZE, SAMPLE_SIZE))
            processed_frames.append(frame_resized)
        
        if model_type == "onnx":
            # ONNX format: [1, 16, 3, 112, 112]
            frames_array = np.stack(processed_frames)  # [16, 112, 112, 3]
            frames_array = frames_array.transpose(0, 3, 1, 2)  # [16, 3, 112, 112]
            frames_array = np.expand_dims(frames_array, axis=0)  # [1, 16, 3, 112, 112]
            frames_array = frames_array.astype(np.float32) / 255.0
            return frames_array
        else:
            # PyTorch format: [1, 16, 3, 112, 112]
            frames_tensor = []
            for frame in processed_frames:
                frame_tensor = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
                frames_tensor.append(frame_tensor)
            frames_tensor = torch.stack(frames_tensor)  # [16, 3, 112, 112]
            return frames_tensor.unsqueeze(0)  # [1, 16, 3, 112, 112]
    except Exception as e:
        print(f"Error preprocessing frames: {e}")
        return None

# WhatsApp config
PHONE_NUMBER = "+237678931432"
last_sent_action = None
last_sent_time = 0
SEND_COOLDOWN = 30  # Increased to 30 seconds
USE_WHATSAPP = True  # Set to False to disable WhatsApp

# Action confidence tracking
action_confidence = {}
MIN_CONFIDENCE_COUNT = 3  # Need 3 consecutive same predictions
current_action_count = 0
current_action = None

def send_whatsapp_message(action):
    global last_sent_action, last_sent_time, USE_WHATSAPP, current_action_count, current_action
    current_time = time.time()
    
    if not USE_WHATSAPP:
        print(f"📱 Would send: {action}")
        return
    
    # Track action confidence
    if action == current_action:
        current_action_count += 1
    else:
        current_action = action
        current_action_count = 1
    
    # Only send if:
    # 1. Action is different from last sent
    # 2. Enough time has passed (30 seconds)
    # 3. Action is not "unknown"
    # 4. We have high confidence (3+ consecutive predictions)
    if (action != last_sent_action and 
        current_time - last_sent_time > SEND_COOLDOWN and
        action != "unknown" and
        current_action_count >= MIN_CONFIDENCE_COUNT):
        try:
            message = f"Action detected: {action} at {datetime.now().strftime('%H:%M')}"
            print(f"📱 Sending: {message}")
            pwk.sendwhatmsg_instantly(PHONE_NUMBER, message, wait_time=10)
            last_sent_action = action
            last_sent_time = current_time
            print(f"✅ Message sent successfully")
        except Exception as e:
            print(f"❌ WhatsApp error: {e}")
            print(f"📱 Would send: {action}")
            # Disable WhatsApp for future calls
            USE_WHATSAPP = False

# Main loop
print("Press 'q' to quit")
print(f"WhatsApp messages will be sent to: {PHONE_NUMBER}")

frame_buffer = []
frame_count = 0
processed_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video")
        break
    
    frame_count += 1
    
    # Add frame to buffer
    frame_buffer.append(frame)
    
    # When we have enough frames, process them
    if len(frame_buffer) >= SAMPLE_DURATION:
        # Get the last 16 frames
        frames_to_process = frame_buffer[-SAMPLE_DURATION:]
        
        # Predict action
        if model is not None:
            try:
                input_data = preprocess_frames(frames_to_process, model_type)
                if input_data is not None:
                    if model_type == "onnx":
                        # ONNX inference
                        input_name = model.get_inputs()[0].name
                        output_name = model.get_outputs()[0].name
                        outputs = model.run([output_name], {input_name: input_data})
                        prediction = np.array(outputs[0])
                        predicted_class = np.argmax(prediction, axis=1)[0]
                    else:
                        # PyTorch inference
                        with torch.no_grad():
                            prediction = model(input_data)
                            predicted_class = torch.argmax(prediction, dim=1).item()
                    
                    predicted_class = int(predicted_class)
                    predicted_action = actions[predicted_class] if predicted_class < len(actions) else "unknown"
                    processed_count += 1
                else:
                    predicted_action = "unknown"
            except Exception as e:
                print(f"Prediction error: {e}")
                predicted_action = "unknown"
        else:
            predicted_action = "no_model"
        
        # Send WhatsApp message
        send_whatsapp_message(predicted_action)
        
        # Display on the last frame (with memory management)
        try:
            display_frame = frames_to_process[-1].copy()
            cv.putText(display_frame, f'Action: {predicted_action}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.putText(display_frame, f'Frame: {frame_count}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv.putText(display_frame, f'Processed: {processed_count}', (10, 110), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv.putText(display_frame, f'Confidence: {current_action_count}/{MIN_CONFIDENCE_COUNT}', (10, 150), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv.putText(display_frame, f'Model: {model_type.upper() if model_type else "NONE"}', (10, 190), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            cv.imshow('3D CNN Action Detection', display_frame)
        except Exception as e:
            print(f"Display error: {e}")
        
        # Clear some frames from buffer to save memory
        if len(frame_buffer) > SAMPLE_DURATION + 5:
            frame_buffer = frame_buffer[-SAMPLE_DURATION:]
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows() 