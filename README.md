# Damaged-road-sign-detection-Australian-dataset-
Detection of damaged road sign 
Traffic Sign Image Processing Project
Overview
This project focuses on processing images of traffic signs to simulate realistic degradation effects such as blending, pigmentation, and rusting. The workflow involves:

Cropping traffic signs from full images.
Applying image processing techniques to simulate degradation (bending, pigmentation, rusting).
Reconstructing the processed traffic signs back into the original image.

The provided code implements these steps using Python and OpenCV (cv2).
Project Structure

crop_traffic_sign.py: Script to detect and crop traffic signs from full images.
apply_effects.py: Script to apply bending, pigmentation, and rusting effects to cropped traffic sign images.
reconstruct_image.py: Script to reconstruct the full image by placing processed traffic signs back into their original positions.
images/: Directory containing input images (e.g., full traffic scene images).
rust_textures/: Directory containing rust texture images for the rusting effect.
output/: Directory to save cropped, processed, and reconstructed images.

Prerequisites

Python 3.x
Required libraries:pip install opencv-python numpy


Input images of traffic scenes containing traffic signs.
Rust texture images for the rusting effect.

Workflow
1. Cropping Traffic Signs

Purpose: Extract traffic signs from full images to isolate them for processing.
Method:
Use contour detection or a pre-trained model (e.g., Haar- Code: crop_traffic_sign.py


Output: Cropped traffic sign images saved in the output/ directory.

2. Applying Image Processing Techniques

Purpose: Simulate degradation effects on cropped traffic sign images.
Techniques:
Blending: Combine the traffic sign with a rust texture using alpha blending to simulate rust.
Pigmentation: Adjust color channels to mimic fading or discoloration.
Rusting: Overlay random rust patches using a rust texture image and contour-based masking.


Code: apply_effects.py
Output: Processed traffic sign images with degradation effects, saved in the output/ directory.

3. Reconstructing the Image

Purpose: Place processed traffic signs back into the original full image.
Method:
Use the coordinates from the cropping step to paste the processed signs into their original positions.


Code: reconstruct_image.py
Output: Reconstructed full images with degraded traffic signs, saved in the output/ directory.

Usage

Prepare Input:
Place full images in the images/ directory.
Place rust texture images in the rust_textures/ directory.


Run Cropping:python cropped.py


Apply Effects:python.py


Reconstruct Images:python reconstruct_image.py


View Results:
Check the output/ directory for cropped, processed, and reconstructed images.



Example Code
Cropping Example (crop_traffic_sign.py)
import cv2
import numpy as np

# Load full image
image = cv2.imread("images/traffic_scene.jpg")
# Detect and crop traffic sign (simplified example)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    cropped = image[y:y+h, x:x+w]
    cv2.imwrite(f"output/traffic_sign_{i}.jpg", cropped)

Rusting Example (apply_effects.py)
import cv2
import numpy as np

# Load cropped traffic sign and rust texture
image = cv2.imread("output/traffic_sign_0.jpg")
rust = cv2.imread("rust_textures/rust_texture.jpg")
image = cv2.resize(image, (512, 512))
rust = cv2.resize(rust, (512, 512))

# Create random rust patches
blob_mask = np.zeros((512, 512), dtype=np.uint8)
for _ in range(10):
    center = (np.random.randint(50, 462), np.random.randint(50, 462))
    axes = (np.random.randint(30, 80), np.random.randint(30, 80))
    angle = np.random.randint(0, 180)
    cv2.ellipse(blob_mask, center, axes, angle, 0, 360, 255, -1)

# Apply rust effect
contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
overlay = image.copy()
alpha = 0.6
for cnt in contours:
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [cnt], -1, (255, 255, 255), -1)
    blended = cv2.addWeighted(image, 1 - alpha, rust, alpha, 0)
    overlay = np.where(mask == 255, blended, overlay)

cv2.imwrite("output/traffic_sign_rusted.jpg", overlay)

Reconstruction Example (reconstruct_image.py)
import cv2

# Load original and processed images
original = cv2.imread("images/traffic_scene.jpg")
processed = cv2.imread("output/traffic_sign_rusted.jpg")

# Assume coordinates from cropping (example)
x, y, w, h = 100, 150, 512, 512  # Replace with actual coordinates
original[y:y+h, x:x+w] = processed

cv2.imwrite("output/reconstructed_image.jpg", original)

Notes

Ensure file paths in the code are updated to match your directory structure.
The rusting effect requires a valid rust texture image. Update the path in apply_effects.py accordingly.
The cropping code is a simplified example. For robust traffic sign detection, consider using a pre-trained model (e.g., YOLO, Haar cascades).
Adjust parameters (e.g., alpha, number of blobs) in apply_effects.py to control the intensity of degradation effects.

Future Improvements

Implement advanced traffic sign detection using deep learning models.
Add support for multiple traffic signs in a single image.
Introduce additional degradation effects (e.g., scratches, dirt).

License
This project is licensed under the MIT License.

