import streamlit as st
from deepface import DeepFace
import tempfile
import os
import pandas as pd
from datetime import datetime
import random

st.set_page_config(page_title="Facial Recognition Fraud Demo", layout="centered")
st.title("🧠 Facial Recognition Fraud Detection")

st.markdown("Simulate a facial recognition check at a bank. Can an imposter pass?")
st.sidebar.header("🧪 Upload Photos for Comparison")

# Upload images
img1 = st.sidebar.file_uploader("Upload Reference Face (On File)", type=["jpg", "jpeg", "png"])
img2 = st.sidebar.file_uploader("Upload Input Face (Login Attempt)", type=["jpg", "jpeg", "png"])

# Optional: Spoof detection checkbox
show_spoof = st.sidebar.checkbox("Show Spoof Confidence Bar")
log_attempts = st.sidebar.checkbox("Log Fraud Attempts")

# Handle comparison
if img1 and img2:
    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Reference Face", use_column_width=True)
    with col2:
        st.image(img2, caption="Input Face", use_column_width=True)

    # Save images temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f1:
        f1.write(img1.read())
        img1_path = f1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f2:
        f2.write(img2.read())
        img2_path = f2.name

    try:
        st.subheader("🔍 Verification Result")
        result = DeepFace.verify(img1_path, img2_path, model_name='VGG-Face', enforce_detection=False)
        verified = result["verified"]
        distance = result["distance"]
        threshold = result["threshold"]

        st.metric("Distance", f"{distance:.2f}", delta=f"Threshold: {threshold:.2f}")
        st.write(f"Match Result: **{verified}**")

        if verified:
            st.success("✅ Identity Verified - Access Granted")
        else:
            st.error("❌ Identity Mismatch - Fraud Detected")

        # Spoof confidence bar (simulated for now)
        if show_spoof:
            st.subheader("🕵️‍♀️ Spoof Confidence")
            spoof_score = random.uniform(0.1, 0.95)
            st.progress(spoof_score)
            if spoof_score > 0.7:
                st.warning("⚠️ High Spoof Confidence - Possible Photo or Mask Attack")

        # Log attempts if fraud
        if log_attempts and not verified:
            log = {
                "timestamp": datetime.now().isoformat(),
                "verified": verified,
                "distance": distance,
                "ref_file": img1.name,
                "input_file": img2.name,
            }
            log_df = pd.DataFrame([log])
            if os.path.exists("fraud_log.csv"):
                log_df.to_csv("fraud_log.csv", mode="a", header=False, index=False)
            else:
                log_df.to_csv("fraud_log.csv", index=False)
            st.info("🔒 Fraud attempt logged.")

    except Exception as e:
        st.error("Face detection failed. Try different images.")
        st.exception(e)

    # Clean up
    os.remove(img1_path)
    os.remove(img2_path)
else:
    st.info("Please upload two face images to begin.")
