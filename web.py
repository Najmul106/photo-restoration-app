import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pipeline import PhotoRestorationPipeline

# 1. Configure the Web Page layout
st.set_page_config(page_title="Historical Photo Restoration", layout="wide")

# 2. Load the Pipeline (Cached so it doesn't reload heavy AI models on every click)
@st.cache_resource
def load_pipeline():
    return PhotoRestorationPipeline()

pipeline = load_pipeline()

# 3. Build the UI Header
st.title("Historical Photo Restoration & Defect Detection")
st.markdown("""
Upload a damaged historical photo to run it through a custom computer vision pipeline. 
The system uses **YOLOv8** to detect scratches/tears, **OpenCV** to structurally repair the image, 
and **PyTorch** for deep learning colorization.
""")

# 4. Build the Sidebar Controls
st.sidebar.header("Pipeline Settings")
conf_threshold = st.sidebar.slider("Damage Detection Confidence", 0.0, 1.0, 0.25, 0.05, 
                                   help="Lower values detect more defects but may cause false positives.")
inpaint_radius = st.sidebar.slider("Inpaint Repair Radius", 1, 10, 3, 1,
                                   help="How far outside the mask OpenCV should look to borrow repair pixels.")

st.sidebar.header("Active Stages")
run_detection = st.sidebar.checkbox("1. Defect Detection (YOLO)", value=True)
run_repair = st.sidebar.checkbox("2. Structural Repair (Inpainting)", value=True)
run_color = st.sidebar.checkbox("3. Colorization (PyTorch)", value=True)

# 5. Build the Main Workspace
uploaded_file = st.file_uploader("Choose a photo to restore...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded web file into an OpenCV image array
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_img = cv2.imdecode(file_bytes, 1)
    
    st.subheader("Original Image")
    # OpenCV uses BGR, but Streamlit/Web uses RGB, so we convert it for display
    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=False, width=600)
    
    # 6. Run the Pipeline on Button Click
    if st.button("Run Restoration Pipeline", type="primary"):
        with st.spinner("Processing image through AI pipeline..."):
            current_img = original_img.copy()
            
            # --- STEP 1: DETECTION ---
            if run_detection:
                st.write("---")
                st.subheader("Step 1: Defect Detection (YOLO)")
                mask, results = pipeline.generate_inpainting_mask(current_img, confidence_threshold=conf_threshold)
                
                # Display Mask and YOLO Bounding Boxes side-by-side
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Generated Binary Mask**")
                    st.caption("White areas map the exact pixels to be deleted and repaired.")
                    st.image(mask, use_container_width=True)
                
                with col2:
                    st.write("**YOLO Detections**")
                    st.caption("Bounding boxes identifying the damage.")
                    # Get the plotted image straight from YOLO results
                    annotated_img = results[0].plot() 
                    st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                # If skipped, generate a blank mask so the next steps don't break
                mask = np.zeros(current_img.shape[:2], dtype=np.uint8)
            
            # --- STEP 2: INPAINTING ---
            if run_repair:
                st.write("---")
                st.subheader("Step 2: Structural Repair (OpenCV Inpainting)")
                repaired_img = pipeline.apply_inpainting(current_img, mask, inpaint_radius=inpaint_radius)
                st.image(cv2.cvtColor(repaired_img, cv2.COLOR_BGR2RGB), use_container_width=False, width=600)
                current_img = repaired_img
                
            # --- STEP 3: COLORIZATION ---
            if run_color:
                st.write("---")
                st.subheader("Step 3: Deep Colorization (PyTorch)")
                colorized_img = pipeline.apply_colorization(current_img)
                st.image(cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB), use_container_width=False, width=600)
                current_img = colorized_img
                
            st.success("Pipeline processing completed successfully!")
            st.balloons()