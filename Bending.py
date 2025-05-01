import cv2
import numpy as np
import os
from glob import glob
from google.colab.patches import cv2_imshow

# === CONFIG ===
input_dir = "/content/bbox_crops"  # Folder of cropped bbox images
output_dir = "/content/output_bent_images"
os.makedirs(output_dir, exist_ok=True)

def apply_realistic_cylinder_bend(img, curvature=0.0008):
    h, w = img.shape[:2]

    # Map coordinates
    map_y, map_x = np.indices((h, w), dtype=np.float32)

    # Apply cylindrical bend (horizontal bending)
    dx = map_x - w / 2
    bend = curvature * dx**2  # parabolic shape for cylindrical bend
    map_y += bend.astype(np.float32)

    # Remap image
    bent = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return bent

# === PROCESS ALL IMAGES ===
image_paths = glob(os.path.join(input_dir, "*.jpg"))

for img_path in image_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Could not read: {img_path}")
        continue

    bent_img = apply_realistic_cylinder_bend(img, curvature=0.0012)  # Increase curvature for stronger bend
    output_path = os.path.join(output_dir, os.path.basename(img_path))
    cv2.imwrite(output_path, bent_img)
    print(f"✅ Saved bent image: {output_path}")
