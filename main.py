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
    Prepare a PIL Image for ModNet ONNX inference by converting to RGB, resizing, normalizing, and arranging axes.
    
    Parameters:
        image (PIL.Image.Image): Input image; will be converted to RGB if necessary.
        size (int): Target square size (pixels) for both width and height; default is 512.
    
    Returns:
        numpy.ndarray: Float32 array with shape (1, 3, size, size), values in [0.0, 1.0], in CHW order with a leading batch dimension.
    """
    im = image.convert("RGB").resize((size, size), Image.BILINEAR)
    im = np.array(im).astype(np.float32) / 255.0
    im = np.transpose(im, (2, 0, 1))  # HWC -> CHW
    im = np.expand_dims(im, 0)        # add batch dim
    return im

def postprocess(mask, orig_size):
    """
    Convert a model output tensor into a 2D uint8 alpha mask resized to the original image size.
    
    Parameters:
        mask (np.ndarray): Model prediction tensor or array (batch/channel dimensions allowed) containing raw scores.
        orig_size (tuple[int, int]): Target size as (width, height) in pixels for the output mask (OpenCV dsize ordering).
    
    Returns:
        np.ndarray: 2D uint8 mask in range [0, 255] resized to orig_size.
    """
    mask = mask.squeeze()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)  # normalize
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_LINEAR)
    return mask

def remove_bg_modnet(image: Image.Image, ort_sess: ort.InferenceSession):
    """
    Remove the background from a PIL image using a ModNet ONNX inference session and return an RGBA image.
    
    Runs the provided ONNX Runtime session to predict a foreground alpha mask for the input image, postprocesses that mask to the original image size, and composes it as the alpha channel of the original image (converted to RGB). The returned image is a PIL Image in RGBA mode where the alpha channel encodes foreground transparency produced by ModNet.
    
    Parameters:
        image (PIL.Image.Image): Input image to process.
    
    Returns:
        PIL.Image.Image: RGBA image with the predicted alpha mask applied as the alpha channel.
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
    Process an input image by removing its background, compositing the foreground onto a blue background, and (if a face is detected) cropping to the face region with padding.
    
    The function:
    - Opens the provided image.
    - Removes the background using the ModNet ONNX pipeline and pastes the resulting RGBA foreground onto a preloaded blue background resized to the input image size.
    - Runs Mediapipe face detection on the composited image; if at least one face is detected it crops the image to the first detection's bounding box plus 40% padding in both width and height (coordinates clamped to image bounds).
    - Returns the composed (and possibly cropped) PIL.Image.
    
    Parameters:
        image: A path-like object or file-like object accepted by PIL.Image.open representing the image to process.
    
    Returns:
        PIL.Image.Image: The processed image (composited on the blue background and cropped to the face region when detected).
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
    Display a Streamlit UI to upload images, run batch processing, and show/save the processed results.
    
    This function builds a simple Streamlit app flow:
    - Ensures a session-state flag ('button_state') exists to enable/disable the Process button.
    - Shows a title and a file uploader that accepts multiple JPG/PNG images.
    - When files are uploaded, enables a "Process images" button. While processing, a spinner is shown.
    - If the user clicks "Process images", an optional "Stop" button can re-disable processing.
    - Processes each uploaded file with process_image(), displays the resulting image in a three-column layout (round-robin), and saves each processed image to the local "output/" directory using the original filename.
    
    Side effects:
    - Writes output image files to the "output/" folder.
    - Modifies Streamlit session state ('button_state') and renders UI elements.
    
    Returns:
    - None
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
