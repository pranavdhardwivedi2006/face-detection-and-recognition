import streamlit as st
import cv2
import numpy as np
import os
import json
from PIL import Image
import uuid

# --- Setup UI ---
st.set_page_config(page_title="Visionary - Face Recognition", layout="centered")

# --- True Neon Cyberpunk CSS ---
true_neon_css = """
<style>
    /* Deep Black Background */
    .stApp {
        background-color: #050505;
        color: #ffffff;
    }
    
    /* Sidebar - Dark transparent */
    [data-testid="stSidebar"] {
        background-color: #0a0a0a !important;
        border-right: 2px solid #ff00ff;
        box-shadow: 5px 0 15px rgba(255, 0, 255, 0.2);
    }

    /* Glowing Titles */
    h1, h2, h3 {
        color: #00ffff !important;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Standard Text & Labels */
    p, label, .stMarkdown {
        color: #e0e0e0 !important;
        font-weight: bold;
        letter-spacing: 1px;
    }

    /* Hollow Neon Buttons */
    div.stButton > button {
        background-color: transparent !important;
        color: #ff00ff !important;
        font-weight: 900;
        border: 2px solid #ff00ff;
        border-radius: 5px;
        padding: 10px 24px;
        box-shadow: 0 0 10px rgba(255, 0, 255, 0.5), inset 0 0 10px rgba(255, 0, 255, 0.5);
        transition: all 0.2s ease-in-out;
        text-transform: uppercase;
        letter-spacing: 2px;
        width: 100%;
    }

    /* Button Hover - Fill with Neon */
    div.stButton > button:hover {
        background-color: #ff00ff !important;
        color: #050505 !important;
        box-shadow: 0 0 20px #ff00ff, inset 0 0 20px #ff00ff !important;
    }

    /* Input Fields (Cyan Glow) */
    .stNumberInput > div > div > input, .stTextInput > div > div > input, .stSelectbox > div > div {
        background-color: #0f0f0f !important;
        color: #00ffff !important;
        border: 1px solid #00ffff !important;
        border-radius: 5px;
        box-shadow: 0 0 8px rgba(0, 255, 255, 0.3);
    }
    
    /* Focus state for inputs */
    .stNumberInput > div > div > input:focus, .stTextInput > div > div > input:focus {
        box-shadow: 0 0 15px #00ffff !important;
    }
    
    /* Progress bar neon cyan */
    .stProgress > div > div > div > div {
        background-color: #00ffff;
        box-shadow: 0 0 15px #00ffff;
    }
</style>
"""
st.markdown(true_neon_css, unsafe_allow_html=True)

st.title("Vision AI")
st.markdown("### Face Detection and Recognition System")

# Create dataset directory if not exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Sidebar Navigation
menu = ["Add Face Data", "Train Model", "Live Recognition"]
choice = st.sidebar.selectbox("System Module", menu)

# --- Module 1: Add Face Data ---
if choice == "Add Face Data":
    st.subheader("Stage 1: Harvest Data")
    
    # --- 1. State Machine Initialization ---
    if 'capture_status' not in st.session_state:
        st.session_state.capture_status = 'idle'  # states: 'idle', 'running', 'completed'
    if 'input_key_counter' not in st.session_state:
        st.session_state.input_key_counter = 0

    # Dynamic key for resetting the input field
    dynamic_key = f"user_name_input_{st.session_state.input_key_counter}"
    user_name = st.text_input("Enter User Name (e.g., Grand Master):", key=dynamic_key)

    # Layout: Col1 (Left) for Start button, Col2 (Right) for Done button (Red Circle area)
    col1, col2 = st.columns([1, 1])

    # ==========================================
    # STATE: IDLE (Waiting to start)
    # ==========================================
    if st.session_state.capture_status == 'idle':
        with col1:
            if st.button("Start Collecting Data"):
                if not user_name.strip():
                    st.error("Error: Please enter a valid name first.")
                else:
                    # Switch state to running and refresh UI
                    st.session_state.capture_status = 'running'
                    st.rerun()

    # ==========================================
    # STATE: RUNNING (Camera is open)
    # ==========================================
    elif st.session_state.capture_status == 'running':
        # --- Auto-Generate UUID ---
        raw_uuid = uuid.uuid4()
        user_id = raw_uuid.int % 1000000 
        
        # --- Save Name to JSON ---
        names_dict = {}
        if os.path.exists("names.json"):
            with open("names.json", "r") as f:
                try: names_dict = json.load(f)
                except json.JSONDecodeError: pass
        
        names_dict[str(user_id)] = user_name
        with open("names.json", "w") as f:
            json.dump(names_dict, f)

        # --- Face Capture Logic ---
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        cap = cv2.VideoCapture(0)
        
        # Ye placeholders sirf 'running' state me exist karte hain
        frame_placeholder = st.empty()
        progress_bar = st.progress(0)
        
        count = 0
        num_samples = 50
        
        while count < num_samples:
            ret, frame = cap.read()
            if not ret:
                st.error("Camera connection lost.")
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                count += 1
                cropped_face = gray[y:y+h, x:x+w]
                cv2.imwrite(f"dataset/User.{user_id}.{count}.jpg", cropped_face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 3)
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")
            progress_bar.progress(int((count / num_samples) * 100))
            
        cap.release()
        
        # Camera band hone ke baad status 'completed' kar do aur UI refresh karo
        st.session_state.capture_status = 'completed'
        st.rerun()

    # ==========================================
    # STATE: COMPLETED (Waiting for 'Done' click)
    # ==========================================
    elif st.session_state.capture_status == 'completed':
        st.success(f"Target Acquired: '{user_name}' successfully registered in the neural database.")
        
        # col2 uss exact location par hai jahan tumne red mark kiya tha
        with col2: 
            if st.button("Done", key="done_btn"):
                # Reset ALL states back to square one
                st.session_state.capture_status = 'idle'
                st.session_state.input_key_counter += 1
                st.rerun()

# --- Module 2: Train Model ---
elif choice == "Train Model":
    st.subheader("Stage 2: Train LBPH Engine")
    
    if st.button("Initialize Training Sequence"):
        with st.spinner("Compiling facial geometry..."):
            try:
                recognizer = cv2.face.LBPHFaceRecognizer_create()
            except AttributeError:
                st.error("Missing dependency! Run: pip install opencv-contrib-python")
                st.stop()
                
            image_paths = [os.path.join('dataset', f) for f in os.listdir('dataset') if f.endswith('.jpg')]
            
            if len(image_paths) == 0:
                st.warning("No data found! Go to 'Add Face Data' first.")
                st.stop()
                
            faces, ids = [], []
            
            for image_path in image_paths:
                img = Image.open(image_path).convert('L')
                img_numpy = np.array(img, 'uint8')
                try:
                    current_id = int(os.path.split(image_path)[-1].split(".")[1])
                except (IndexError, ValueError):
                    continue
                    
                faces.append(img_numpy)
                ids.append(current_id)
                
            ids = np.array(ids, dtype=np.int32)
            recognizer.train(faces, ids)
            recognizer.write('trainer.yml')
            
            st.success(f"Successfully trained on {len(np.unique(ids))} unique IDs!")

# --- Module 3: Live Recognition ---
elif choice == "Live Recognition":
    st.subheader("Stage 3: Live Recognition Feed")
    
    run_camera = st.checkbox("Initialize Camera Link")
    frame_placeholder = st.empty()
    
    if run_camera:
        if not os.path.exists('trainer.yml'):
            st.error("Model not found! Please train the model first.")
        else:
            # --- Load Names from JSON dynamically ---
            names_dict = {}
            if os.path.exists("names.json"):
                with open("names.json", "r") as f:
                    try:
                        names_dict = json.load(f)
                    except json.JSONDecodeError:
                        names_dict = {}
            # ----------------------------------------

            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainer.yml')
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            
            cap = cv2.VideoCapture(0)
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame.")
                    break
                    
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(30, 30))
                
                for (x, y, w, h) in faces:
                    id_predicted, distance = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if distance < 80:
                        # Fetch name dynamically (keys in JSON are stored as strings)
                        name = names_dict.get(str(id_predicted), f"ID {id_predicted}")
                        confidence = f"  {round(100 - distance)}%"
                        color = (0, 255, 255) # Neon Cyan for recognized
                    else:
                        name = "Unknown Target"
                        confidence = f"  {round(100 - distance)}%"
                        color = (0, 0, 255) # Red for unknown
                        
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Draw a solid background box for the text to make it pop
                    cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)
                    cv2.putText(frame, name, (x+5, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,0,0), 2)
                    cv2.putText(frame, confidence, (x+5, y+h-10), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")
                
            cap.release()