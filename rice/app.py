# app.py
# Run with: streamlit run app.py

import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array


from PIL import Image
import hashlib
import datetime
from fpdf import FPDF
import csv
import io

# Constants
DATASET_DIR = 'images/'
MODEL_PATH = 'rice_classifier_model.h5'
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.7  # If below this, consider unknown but show closest

import hashlib
import os

# Global or passed in as needed
DATASET_DIR = "/path/to/dataset"

def compute_dataset_hash():
    hasher = hashlib.md5()

    for root, dirs, files in os.walk(DATASET_DIR):
        for filename in sorted(files):
            file_path = os.path.join(root, filename)
            with open(file_path, 'rb') as f:
                # Read in chunks to handle large files efficiently
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)

    return hasher.hexdigest()


# Check if model needs retraining (simple check: if hash file exists and matches)
HASH_FILE = 'dataset_hash.txt'
current_hash = compute_dataset_hash()
if os.path.exists(HASH_FILE):
    with open(HASH_FILE, 'r') as f:
        saved_hash = f.read()
    if saved_hash != current_hash:
        st.warning("Dataset changed. Please run train_model.py to retrain the model.")
else:
    st.warning("No hash found. Please run train_model.py to train the model.")
with open(HASH_FILE, 'w') as f:
    f.write(current_hash)

# Load model
@st.cache_resource
def load_rice_model():
    return load_model(MODEL_PATH)

model = load_rice_model()
classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])

# Preprocessing function
def preprocess_image(image):
    # Convert to OpenCV
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Noise reduction
    img_cv = cv2.GaussianBlur(img_cv, (5, 5), 0)
    # Edge detection (optional, for highlighting grains)
    edges = cv2.Canny(img_cv, 100, 200)
    # But for classification, use original blurred
    img_resized = cv2.resize(img_cv, IMG_SIZE)
    img_array = img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, image  # Return processed for ML, original for display

# Classification function
def classify_rice(image_array):
    predictions = model.predict(image_array)
    confidence = np.max(predictions)
    class_idx = np.argmax(predictions)
    rice_type = classes[class_idx]
    if confidence < CONFIDENCE_THRESHOLD:
        return f"Unknown (closest match: {rice_type})", confidence * 100
    return rice_type, confidence * 100

# Find closest match image (simple: pick first from class folder)
def get_closest_match_image(rice_type):
    class_dir = os.path.join(DATASET_DIR, rice_type)
    if os.path.exists(class_dir):
        files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
        if files:
            return os.path.join(class_dir, files[0])
    return None

# Main UI
st.title("Rice Scanning and Classification System")

# Sidebar for navigation
page = st.sidebar.selectbox("Navigate", ["Classify Rice", "Browse Database"])

if page == "Classify Rice":
    st.header("Upload or Capture Rice Image")
    
    # Input options
    upload_option = st.radio("Input Method", ("File Upload", "Camera Capture"))
    
    uploaded_image = None
    if upload_option == "File Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            uploaded_image = Image.open(uploaded_file)
    else:
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            uploaded_image = Image.open(camera_image)
    
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocess and classify
        processed_image, original_image = preprocess_image(uploaded_image)
        rice_type, confidence = classify_rice(processed_image)
        
        # Display results
        st.subheader("Classification Result")
        st.write(f"This is {confidence:.2f}% similar to {rice_type} Rice.")
        
        # Side-by-side comparison
        if st.checkbox("Show Side-by-Side Comparison"):
            closest_image_path = get_closest_match_image(rice_type.split(" (closest match: ")[0] if "Unknown" in rice_type else rice_type)
            if closest_image_path:
                closest_image = Image.open(closest_image_path)
                col1, col2 = st.columns(2)
                col1.image(original_image, caption="Uploaded Image", use_column_width=True)
                col2.image(closest_image, caption=f"Closest Match: {rice_type}", use_column_width=True)
        
        # Export options
        st.subheader("Export Report")
        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_data = {
            "Rice Type": rice_type,
            "Confidence": f"{confidence:.2f}%",
            "Date": date_time
        }
        
        if st.button("Download PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for key, value in report_data.items():
                pdf.cell(200, 10, txt=f"{key}: {value}", ln=1)
            pdf_output = io.BytesIO()
            pdf.output(dest='S').encode('latin1')  # Save to bytes
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            st.download_button("Download PDF", pdf_bytes, file_name="rice_report.pdf", mime="application/pdf")
        
        if st.button("Download CSV"):
            csv_output = io.StringIO()
            writer = csv.DictWriter(csv_output, fieldnames=report_data.keys())
            writer.writeheader()
            writer.writerow(report_data)
            st.download_button("Download CSV", csv_output.getvalue(), file_name="rice_report.csv", mime="text/csv")

elif page == "Browse Database":
    st.header("Searchable Rice Database")
    search_term = st.text_input("Search for Rice Type")
    for rice_type in classes:
        if search_term.lower() in rice_type.lower():
            st.subheader(rice_type)
            class_dir = os.path.join(DATASET_DIR, rice_type)
            files = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
            if files:
                sample_image = Image.open(os.path.join(class_dir, files[0]))
                st.image(sample_image, caption=f"Sample of {rice_type}", width=200)
            st.write(f"Number of images: {len(files)}")

# Note on updating dataset
st.sidebar.info("To add new rice types, create a new subfolder in 'images/' and add images. Then run train_model.py to update the model.")

# Future improvement note
st.sidebar.info("Future: Detect rice quality (broken grains, etc.) - Not implemented yet.")