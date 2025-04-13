import streamlit as st
from deepface import DeepFace
from PIL import Image
import pandas as pd
import tempfile
import os
from datetime import datetime

st.set_page_config(page_title="Facial Match Check", layout="centered")
st.title("🔒 Facial Identity Check - Advai Demo")

st.markdown("Upload a reference (official) photo and a selfie to verify identity.")

# Upload images
ref_img = st.file_uploader("📷 Upload Reference Image", type=["jpg", "png", "jpeg"])
selfie_img = st.file_uploader("🤳 Upload Selfie Image", type=["jpg", "png", "jpeg"])

log_mismatches = st.checkbox("Log Mismatches")
show_confidence = st.checkbox("Show Spoof Confidence (experimental)")

if ref_img and selfie_img:
    with tempfile.NamedTemporaryFile(delete=False) as ref_temp:
        ref_path = ref_temp.name
        ref_temp.write(ref_img.read())

    with tempfile.NamedTemporaryFile(delete=False) as selfie_temp:
        selfie_path = selfie_temp.name
        selfie_temp.write(selfie_img.read())

    st.image([ref_path, selfie_path], caption=["Reference", "Selfie"], width=200)

    try:
        result = DeepFace.verify(ref_path, selfie_path, model_name='VGG-Face', enforce_detection=True)

        if result["verified"]:
            st.success("✅ Identity Match Confirmed")
        else:
            st.error("🚨 Identity Mismatch Detected")
            if log_mismatches:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "match": False,
                    "distance": result["distance"]
                }
                df = pd.DataFrame([log_entry])
                if os.path.exists("mismatch_log.csv"):
                    df_existing = pd.read_csv("mismatch_log.csv")
                    df = pd.concat([df_existing, df], ignore_index=True)
                df.to_csv("mismatch_log.csv", index=False)
                st.info("📄 Mismatch attempt logged.")

        if show_confidence:
            confidence_score = max(0, 1 - result["distance"])  # crude inverse of distance
            st.metric("Confidence (lower = spoof risk)", f"{confidence_score:.2f}")

    except Exception as e:
        st.warning(f"Could not process faces: {e}")

    
               
    
              

