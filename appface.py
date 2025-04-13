import streamlit as st
import face_recognition
from PIL import Image
import numpy as np
import pandas as pd
import io

st.set_page_config(page_title="Facial Fraud Detection", layout="centered")
st.title("🧠 Facial Identity Match Check")

st.write("Upload two face images to check if they're the same person.")

log_file = "fraud_log.csv"
if "log_df" not in st.session_state:
    if not st.session_state.get("log_df_loaded", False) and not st.session_state.get("log_df_failed", False):
        try:
            st.session_state["log_df"] = pd.read_csv(log_file)
        except:
            st.session_state["log_df"] = pd.DataFrame(columns=["Image1", "Image2", "Match"])
        st.session_state["log_df_loaded"] = True

# Upload interface
img1_file = st.file_uploader("Upload First Face", type=["jpg", "jpeg", "png"])
img2_file = st.file_uploader("Upload Second Face", type=["jpg", "jpeg", "png"])

if img1_file and img2_file:
    img1 = Image.open(img1_file)
    img2 = Image.open(img2_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Image 1", use_column_width=True)
    with col2:
        st.image(img2, caption="Image 2", use_column_width=True)

    if st.button("🔍 Compare"):
        try:
            img1_np = face_recognition.load_image_file(io.BytesIO(img1_file.getvalue()))
            img2_np = face_recognition.load_image_file(io.BytesIO(img2_file.getvalue()))

            encoding1 = face_recognition.face_encodings(img1_np)
            encoding2 = face_recognition.face_encodings(img2_np)

            if len(encoding1) == 0 or len(encoding2) == 0:
                st.error("Couldn't detect a face in one of the images.")
            else:
                match = face_recognition.compare_faces([encoding1[0]], encoding2[0])[0]
                distance = np.linalg.norm(encoding1[0] - encoding2[0])

                st.success("✅ Match!" if match else "❌ Not a Match")
                st.metric("Distance", round(distance, 3))

                new_row = {"Image1": img1_file.name, "Image2": img2_file.name, "Match": match}
                st.session_state["log_df"] = pd.concat(
                    [st.session_state["log_df"], pd.DataFrame([new_row])], ignore_index=True
                )
                st.session_state["log_df"].to_csv(log_file, index=False)

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# Logs in sidebar
if st.sidebar.checkbox("📜 Show Match History"):
    st.sidebar.dataframe(st.session_state["log_df"].tail(10))


    
              

