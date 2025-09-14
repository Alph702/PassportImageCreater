import os
import cv2
import mediapipe as mp
import numpy as np
from rembg import remove
from PIL import Image
import streamlit as st

# Initialize Mediapipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

background = Image.open(r"blue_image.png")

def process_image(image):
    img = Image.open(image)

    fg = remove(img)
    bg_resized = background.resize(img.size)
    bg_resized.paste(fg, (0, 0), fg)

    # Convert PIL image to OpenCV format for detection
    img_cv = np.array(bg_resized)
    # Convert RGB to BGR
    img_cv = img_cv[:, :, ::-1].copy()

    # Convert BGR (OpenCV) → RGB (Mediapipe)
    rgb_frame = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)

    if results.detections:
        # Get the first face found
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img_cv.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                     int(bboxC.width * iw), int(bboxC.height * ih)
        
        # Add proportional padding
        padding_w = int(w * 0.4)
        padding_h = int(h * 0.4)
        x1 = max(0, x - padding_w)
        y1 = max(0, y - padding_h)
        x2 = min(img.width, x + w + padding_w)
        y2 = min(img.height, y + h + padding_h)
        # Crop the image
        bg_resized = bg_resized.crop((x1, y1, x2, y2))

    return bg_resized


def main():
    if 'button_state' not in st.session_state:
        st.session_state.button_state = True

    st.title("Create passport style images")

    images = st.file_uploader(
        "Upload images", accept_multiple_files="directory", type=["jpg", "png"]
    )
    if images:
        st.session_state.button_state = False
        button = st.button("Process images", disabled=st.session_state.button_state)
        with st.spinner("Processing images..."):
            if button:
                if st.button("Stop"):
                    st.session_state.button_state = True
                col1, col2, col3 = st.columns(3)

                for idx, image in enumerate(images):
                    processed_image = process_image(image=image)
                    if idx % 3 == 0:
                        col1.image(processed_image)
                    elif idx % 3 == 1:
                        col2.image(processed_image)
                    else:
                        col3.image(processed_image)
                    processed_image.save("output/" + image.name)
        
                st.success("Images prossesed successful and saved in **output** folder")

main()
