import cv2
import json
import time
import numpy as np
import os
import sys

# --- CONFIGURATION ---
VIDEO_PATH = "demo_video.mp4"
DATA_PATH = "rppg_waveforms.npy"

def load_real_data():
    """Loads the actual rPPG data from the .npy file provided by the ML team"""
    if os.path.exists(DATA_PATH):
        try:
            # allow_pickle=True is required for .npy files containing dictionaries
            data = np.load(DATA_PATH, allow_pickle=True).item()
            # Get the first video's data as a sample stream
            first_video = list(data.keys())[0]
            waveforms = data[first_video]['rppg']
            label = data[first_video]['label']
            return waveforms, label
        except Exception as e:
            print(f"Error reading .npy file: {e}", file=sys.stderr)
    return None, None

def start_inference():
    # 1. Load the real data provided by the ML team
    waveforms, base_score = load_real_data()
    
    # 2. Open the video loop
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"CRITICAL: Video {VIDEO_PATH} not found in backend folder!", file=sys.stderr)
        return

    frame_index = 0
    
    while True:
        ret, frame = cap.read()

        # LOOP LOGIC: Restart video and data when finished
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_index = 0
            continue

        # --- DATA PROCESSING ---
        if waveforms is not None:
          
          
            current_signal = float(waveforms[frame_index % len(waveforms)])
            
          
            hr_val = 70 + (current_signal * 5) 
            att_val = base_score * 100
        else:
         
            current_signal = np.sin(time.time() * 5) * 20 + 50
            hr_val = 72 + np.random.randint(-2, 2)
            att_val = 85

      
        status_text = "Engaged" if att_val > 75 else "Distracted" if att_val > 50 else "Drowsy"

        # --- SEND TO NODE.JS ---
        output = {
            "heart_rate": int(hr_val),
            "status": status_text,
            "rppg": float(current_signal)
        }
    
        print(json.dumps(output), flush=True)

        frame_index += 1
        
        # Match video speed (approx 30 FPS)
        time.sleep(0.03)

    cap.release()

if __name__ == "__main__":
    start_inference()