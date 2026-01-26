import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Deepfake Detector", layout="centered")

st.title("🕵️ Deepfake Image Detection")
st.write("Upload an image to check if it might be a deepfake")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def check_deepfake(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score

if uploaded_file:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    score = check_deepfake(image_np)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write(f"🔍 Sharpness Score: **{score:.2f}**")

    if score < 400:
        st.error("⚠️ This image is likely a DEEPFAKE")
    else:
        st.success("✅ This image looks REAL")
