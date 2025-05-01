import cv2
import os
import numpy as np
#from google.colab.patches import cv2_imshow
from glob import glob

# === CONFIG ===
image_folder = "/home/msi/course/ACS/damageimagebrimbank"  # Folder with all images
label_folder = "/home/msi/course/ACS/brimbank_damaged_lables"  # Folder with all YOLO .txt files

# === OUTPUT FOLDERS ===
bbox_dir = "/home/msi/course/ACS/croppedimages/geneatedimages"
masked_dir = "/home/msi/course/ACS/croppedimages/generatedlabels"
os.makedirs(bbox_dir, exist_ok=True)
os.makedirs(masked_dir, exist_ok=True)

# === GET ALL IMAGE FILES ===
image_paths = sorted(glob(os.path.join(image_folder, "*.*")))
supported_ext = ('.jpg', '.jpeg', '.png')

for image_path in image_paths:
    if not image_path.lower().endswith(supported_ext):
        continue

    # === PARSE IMAGE NAME ===
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_path = os.path.join(label_folder, base_name + ".txt")

    if not os.path.exists(label_path):
        print(f"⚠️ Label not found for {base_name}, skipping.")
        continue

    # === LOAD IMAGE ===
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to read image: {image_path}")
        continue

    h_img, w_img = img.shape[:2]

    # === LOAD YOLO LABELS ===
    with open(label_path, 'r') as f:
        lines = f.readlines()

    if not lines:
        print(f"⚠️ No annotations in {label_path}, skipping.")
        continue

    for idx, line in enumerate(lines):
        class_id, x_center, y_center, width, height = map(float, line.strip().split())

        # Convert to pixel values
        x_center *= w_img
        y_center *= h_img
        width *= w_img
        height *= h_img

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Clip to image dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img, x2), min(h_img, y2)

        # === IMAGE 1: Extract BBOX ===
        bbox_crop = img[y1:y2, x1:x2]
        bbox_filename = f"{bbox_dir}/bbox_object_{base_name}_{idx}.jpg"
        cv2.imwrite(bbox_filename, bbox_crop)

        # === IMAGE 2: Masked Image ===
        img_masked = img.copy()
        img_masked[y1:y2, x1:x2] = (255, 255, 255)
        masked_filename = f"{masked_dir}/image_without_object_{base_name}_{idx}.jpg"
        cv2.imwrite(masked_filename, img_masked)

        print(f"✅ Processed {base_name} [BBOX #{idx}]")

print("\n✅ All images processed and saved.")
