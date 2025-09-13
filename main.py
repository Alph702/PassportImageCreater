import os
from cv2 import CascadeClassifier, cvtColor, COLOR_BGR2GRAY
import numpy as np
from rembg import remove
from PIL import Image
import streamlit as st

background = Image.open(r"blue_image.png")

# Load the Haar cascade file for face detection
face_cascade = CascadeClassifier('haarcascade_frontalface_default.xml')

def process_image(image):
    img = Image.open(image)

    fg = remove(img)
    bg_resized = background.resize(img.size)
    bg_resized.paste(fg, (0, 0), fg)

    # Convert PIL image to OpenCV format for detection
    img_cv = np.array(bg_resized)
    # Convert RGB to BGR
    img_cv = img_cv[:, :, ::-1].copy()

    # Convert to grayscale for face detection
    gray = cvtColor(img_cv, COLOR_BGR2GRAY)
    # Detect faces with adjusted parameters
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)

    if len(faces) > 0:
        # Get the first face found
        (x, y, w, h) = faces[0]
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