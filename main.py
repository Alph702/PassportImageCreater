import os
import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from PIL import Image
import streamlit as st

# Initialize Mediapipe Face Detection
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# Load ONNX model
ort_sess = ort.InferenceSession("modnet.onnx", providers=["CPUExecutionProvider"])

background = Image.open(r"blue_image.png")

def preprocess(image: Image.Image, size=512):
    """
    Prepare an RGB image for ModNet ONNX inference.
    
    Converts the PIL image to RGB, resizes it to (size, size), scales pixel values to [0, 1],
    reorders channels from HWC to CHW, and adds a leading batch dimension.
    
    Parameters:
        image (PIL.Image.Image): Input image.
        size (int, optional): Target square size for the model input (width and height). Defaults to 512.
    
    Returns:
        numpy.ndarray: Float32 array shaped (1, 3, size, size) with values in [0, 1], ready for ONNX inference.
    """
    im = image.convert("RGB").resize((size, size), Image.BILINEAR)
    im = np.array(im).astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))  # HWC -> CHW
    im = np.expand_dims(im, 0)        # add batch dim
    return im

def postprocess(mask, orig_size):
    """
    Convert a raw model output into a 2D uint8 alpha mask resized to the original image size.
    
    Normalizes the input tensor to the [0, 255] range, converts to uint8, and resizes to orig_size using linear interpolation.
    
    Parameters:
        mask (numpy.ndarray): Model output mask (typically a float array with shape like (1, 1, H, W) or (H, W)).
        orig_size (tuple[int, int]): Target size as (width, height) in pixels.
    
    Returns:
        numpy.ndarray: 2D uint8 mask with values in [0, 255] and spatial dimensions equal to orig_size.
    """
    mask = mask.squeeze()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)  # normalize
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_LINEAR)
    return mask

def remove_bg_modnet(image: Image.Image, ort_sess: ort.InferenceSession):
    """
    Return an RGBA PIL.Image where the input image's foreground is preserved as the alpha channel using a ModNet ONNX model.
    
    Preprocesses the input image, runs the provided ONNX Runtime session to produce a foreground mask, postprocesses the mask to the original image size, and combines the mask with the input image as the alpha channel. The returned image is in "RGBA" mode and has the same dimensions as the input.
    
    Parameters:
        image (Image.Image): Input image to extract the foreground from.
    
    Returns:
        Image.Image: RGBA image with the computed alpha mask as the alpha channel.
    """
    orig_size = image.size
    input_tensor = preprocess(image)

    # Run inference
    inputs = {ort_sess.get_inputs()[0].name: input_tensor}
    pred = ort_sess.run(None, inputs)[0]

    # Postprocess mask
    mask = postprocess(pred, orig_size)

    # Apply mask as alpha channel
    image_rgb = image.convert("RGB")
    result_rgb = np.array(image_rgb)

    result_rgba = np.dstack((result_rgb, mask))

    return Image.fromarray(result_rgba)

def process_image(image):
    """
    Compose an input image onto the blue background, run face detection, and return either the face-centered crop (with padding) or the full composite.
    
    The function:
    - Opens `image` with PIL, removes its background using the ModNet ONNX pipeline, and pastes the resulting foreground onto the module-level `background`.
    - Runs Mediapipe face detection on the composed image. If a face is detected, crops the composite to the first detection's bounding box expanded by 40% in both width and height, clamped to image bounds.
    - If no face is detected, returns the full background-composited image at the input image size.
    
    Parameters:
        image: A file path or file-like object accepted by PIL.Image.open (e.g., an UploadedFile).
    
    Returns:
        PIL.Image: The face-centered cropped composite if a face was found; otherwise the full composite image.
    """
    img = Image.open(image)

    fg = remove_bg_modnet(img, ort_sess)
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
    """
    Run the Streamlit UI to upload and batch-process images into passport-style photos.
    
    Displays a file uploader (accepts JPG and PNG), a "Process images" button, and a three-column preview layout. When the user uploads files and clicks "Process images", each file is processed via process_image(image), displayed in one of three columns (round-robin), and saved to the output/ directory with the original filename. A session-state flag ('button_state') is used to disable/enable the processing button; a "Stop" button during processing sets that flag to True to halt further processing. If no files are uploaded the process button remains disabled.
    """
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

if __name__ == '__main__':
    main()
