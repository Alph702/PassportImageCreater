import os
from cv2 import CascadeClassifier, cvtColor, COLOR_BGR2GRAY
import numpy as np
from rembg import remove
from PIL import Image

input_folder = r"images"
output_folder = r"output"
# Create a plain white background
background = Image.open(r"blue_image.png")

# Load the Haar cascade file for face detection
# Make sure 'haarcascade_frontalface_default.xml' is in the same directory as the script
face_cascade = CascadeClassifier('haarcascade_frontalface_default.xml')

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith((".jpg", ".png")):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        img = Image.open(input_path)

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

        
        bg_resized.save(output_path)
        print(f"Processed {file}")

# ✅ Show message when done
print("🎉 All photos have been processed successfully! Check the 'output' folder.")