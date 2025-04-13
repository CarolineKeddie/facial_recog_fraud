import streamlit as st
import pandas as pd
import cv2
from deepface import DeepFace
from PIL import Image
import numpy as np
import os
from datetime import datetime

# -----------------------
# App Setup
# -----------------------

st.title("Facial Recognition Identity Check")
st.write("Upload two face images: one official photo (ID/passport) and one live/selfie attempt.")

# Logging directory
log_file = "fraud_attempts_log.csv"
if not os.path.exists(log_file):
    pd.DataFrame(columns=["timestamp", "match", "spoof_score"]).to_csv(log_file, index=False)

# -----------------------
# Image Upload
# -----------------------

uploaded_id = st.file_uploader("Upload official ID image", type=["jpg", "jpeg", "png"], key="id_img")
uploaded_selfie = st.file_uploader("Upload live/selfie image", type=["jpg", "jpeg", "png"], key="selfie_img")

# -----------------------
# Run Face Verification
# -----------------------

if uploaded_id and uploaded_selfie:
    try:
        id_img = Image.open(uploaded_id).convert('RGB')
        selfie_img = Image.open(uploaded_selfie).convert('RGB')

        # Save temporarily
        id_path = "temp_id.jpg"
        selfie_path = "temp_selfie.jpg"
        id_img.save(id_path)
        selfie_img.save(selfie_path)

        # Run DeepFace verification
        result = DeepFace.verify(img1_path=id_path, img2_path=selfie_path, enforce_detection=False)

        match = result["verified"]
        distance = result["distance"]
        threshold = result["threshold"]

        st.subheader("Face Verification Result:")
        st.write(f"Match: {'✅ Yes' if match else '❌ No'}")
        st.write(f"Distance: {distance:.4f} (threshold: {threshold:.4f})")

        # Spoof detection logic (simple version based on distance score)
        spoof_score = 1 - min(distance / threshold, 1.0)
        st.subheader("Spoof Detection Score")
        st.progress(spoof_score)

        # Log attempt
        attempt_log = pd.DataFrame([{
            "timestamp": datetime.utcnow().isoformat(),
            "match": match,
            "spoof_score": round(spoof_score, 4)
        }])
        attempt_log.to_csv(log_file, mode='a', header=False, index=False)

    except Exception as e:
        st.error(f"Something went wrong: {e}")

       
          
           
