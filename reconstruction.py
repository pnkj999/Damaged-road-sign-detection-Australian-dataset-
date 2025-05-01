import cv2
import os
from glob import glob
#from google.colab.patches import cv2_imshow

# === CONFIG ===
label_dir = "/home/msi/course/ACS/brimbank_damaged_lables"
object_dir = "/home/msi/course/ACS/croppedimages/outputrustingeffect"
masked_dir = "/home/msi/course/ACS/croppedimages/generatedlabels"
output_dir = "/home/msi/course/ACS/croppedimages/reconstructuredimage"
os.makedirs(output_dir, exist_ok=True)

# === PROCESS EACH LABEL FILE ===
label_files = sorted(glob(os.path.join(label_dir, "*.txt")))

for label_path in label_files:
    base = os.path.splitext(os.path.basename(label_path))[0]  # e.g., 9134385...-0

    # Build matching filenames
    object_filename = f"bbox_object_{base}_0.jpg"
    masked_filename = f"image_without_object_{base}_0.jpg"

    object_path = os.path.join(object_dir, object_filename)
    masked_path = os.path.join(masked_dir, masked_filename)

    print(f"\n🔍 Processing: {base}")
    print(f"📦 BBox Image: {object_path}")
    print(f"🖼️ Masked Image: {masked_path}")

    object_img = cv2.imread(object_path)
    masked_img = cv2.imread(masked_path)

    if object_img is None or masked_img is None:
        print(f"⚠️ Skipping {base} due to missing image files.")
        continue

    # Read the first annotation line
    try:
        with open(label_path, 'r') as f:
            line = f.readline().strip()
        class_id, x_center, y_center, width, height = map(float, line.split())
    except Exception as e:
        print(f"⚠️ Error reading label {label_path}: {e}")
        continue

    # Convert bbox from normalized to pixel coords
    h_img, w_img = masked_img.shape[:2]
    x_center *= w_img
    y_center *= h_img
    width *= w_img
    height *= h_img

    x1 = int(x_center - width / 2)
    y1 = int(y_center - height / 2)
    x2 = int(x_center + width / 2)
    y2 = int(y_center + height / 2)

    # Clamp box inside image boundaries
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w_img, x2)
    y2 = min(h_img, y2)

    # Resize and paste
    object_resized = cv2.resize(object_img, (x2 - x1, y2 - y1))
    reconstructed_img = masked_img.copy()
    reconstructed_img[y1:y2, x1:x2] = object_resized

    # Save
    out_path = os.path.join(output_dir, f"reconstructed_{base}.jpg")
    cv2.imwrite(out_path, reconstructed_img)
    print(f"✅ Saved reconstructed image: {out_path}")

    # Optionally visualize one example
    # cv2_imshow(reconstructed_img)
