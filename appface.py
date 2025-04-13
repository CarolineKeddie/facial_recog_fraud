import streamlit as st
from deepface import DeepFace
from PIL import Image
import pandas as pd
import numpy as np
import os
import tempfile

st.set_page_config(page_title="Face Match Fraud Detector", layout="centered")
st.title("🧠 Face Match Fraud Detection")

st.write("Upload two face images to compare and check for identity match.")
log_file = "match_logs.csv"

# Initialize logs
if not os.path.exists(log_file):
    pd.DataFrame(columns=["Image1", "Image2", "Match", "Distance"]).to_csv(log_file, index=False)

img1_file = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"])

if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Image 1", use_column_width=True)
    with col2:
        st.image(img2, caption="Image 2", use_column_width=True)

    if st.button("🔍 Compare Faces"):
        with st.spinner("Analyzing..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp1:
                    img1.save(tmp1.name)
                    path1 = tmp1.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp2:
                    img2.save(tmp2.name)
                    path2 = tmp2.name

                result = DeepFace.verify(img1_path=path1, img2_path=path2, enforce_detection=False)

                distance = result["distance"]
                verified = result["verified"]

                st.success("✅ Match!" if verified else "❌ Not a Match")
                st.metric("Similarity Distance", round(distance, 3))

                # Optional: Add spoof score logic here in future

                # Log the result
                log_df = pd.read_csv(log_file)
                new_row = {
                    "Image1": img1_file.name,
                    "Image2": img2_file.name,
                    "Match": verified,
                    "Distance": distance
                }
                log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
                log_df.to_csv(log_file, index=False)

            except Exception as e:
                st.error(f"Error during face verification: {e}")

# View logs
if st.sidebar.checkbox("📜 Show Log History"):
    logs = pd.read_csv(log_file)
    st.sidebar.dataframe(logs.tail(10))

   
