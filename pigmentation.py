import cv2
import numpy as np
import os
import random

def parse_yolo_label(label_file, img_width, img_height):
    """Parse YOLO label file and convert to pixel coordinates."""
    boxes = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x1 = int(x_center - width / 3)
            y1 = int(y_center - height / 3)
            x2 = int(x_center + width / 3)
            y2 = int(y_center + height / 3)
            boxes.append((x1, y1, x2, y2))
    return boxes

def apply_graffiti(img, boxes):
    """Apply a white paint spill effect to the center of each bounding box with 30% padding and reduced opacity."""
    img_graffiti = img.copy()
    height, width = img.shape[:2]
    paint_color = (0, 0, 0)  # White paint

    for (x1, y1, x2, y2) in boxes:
        # Define the center region with 30% padding
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(0.4 * box_width)
        pad_y = int(0.4 * box_height)

        center_x1 = x1 + pad_x
        center_x2 = x2 - pad_x
        center_y1 = y1 + pad_y
        center_y2 = y2 - pad_y

        # Create a mask for the paint spill
        spill_mask = np.zeros((height, width), dtype=np.uint8)

        # Add splatters mainly in the center region
        for _ in range(30):  
            splat_x = random.randint(center_x1, center_x2)
            splat_y = random.randint(center_y1, center_y2)
            radius = random.randint(3, 8)
            if random.random() > 0.5:  
                cv2.ellipse(spill_mask, (splat_x, splat_y), (radius, radius*4), 0, 0, 360, 255, -1)
                drip_length = random.randint(15, 30)
                cv2.line(spill_mask, (splat_x, splat_y + radius*4), 
                         (splat_x, splat_y + radius*4 + drip_length), 255, 2)
            else:
                cv2.circle(spill_mask, (splat_x, splat_y), radius, 255, -1)

        # Blur the splatters
        spill_mask = cv2.GaussianBlur(spill_mask, (15, 15), 0)
        spill_mask = np.clip(spill_mask, 0, 255).astype(np.uint8)

        # Apply the paint spill with reduced opacity (alpha = 0.4)
        alpha = 0.9 * (spill_mask / 255.0)
        spill_layer = np.zeros_like(img_graffiti)
        spill_layer[spill_mask > 50] = paint_color

        alpha_3d = alpha[:, :, np.newaxis]
        img_graffiti = (img_graffiti * (1 - alpha_3d) + spill_layer * alpha_3d).astype(np.uint8)

    return img_graffiti

def process_images(image_dir, label_dir, output_dir):
    """Process all images in a directory with their corresponding labels."""
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return
    if not os.path.exists(label_dir):
        print(f"Label directory not found: {label_dir}")
        return
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Supported image extensions
    image_extensions = ('.jpg', '.jpeg', '.png')
    
    # Process each image file
    processed_count = 0
    for image_file in os.listdir(image_dir):
        if image_file.lower().endswith(image_extensions):
            # Get base filename without extension
            base_name = os.path.splitext(image_file)[0]
            
            # Construct paths
            image_path = os.path.join(image_dir, image_file)
            label_path = os.path.join(label_dir, f"{base_name}.txt")
            
            # Check if corresponding label file exists
            if not os.path.exists(label_path):
                print(f"Skipping {image_file}: No corresponding label file found")
                continue

            # Load and process image
            img = cv2.imread(image_path)
            if img is None:
                print(f"Failed to load image: {image_path}")
                continue
                
            height, width = img.shape[:2]
            boxes = parse_yolo_label(label_path, width, height)
            
            if not boxes:
                print(f"No bounding boxes found in {label_path}")
                continue

            # Apply graffiti effect
            graffiti_img = apply_graffiti(img, boxes)

            # Save result
            output_path = os.path.join(output_dir, f"{base_name}_graffiti.jpg")
            cv2.imwrite(output_path, graffiti_img)
            
            processed_count += 1
            print(f"Processed {image_file} -> {output_path}")

    print(f"Completed processing {processed_count} images")

# Example usage
image_dir = "/home/msi/course/ACS/damageimagebrimbank"
label_dir = "/home/msi/course/ACS/brimbank_damaged_lables"
output_dir = "/home/msi/course/ACS/road_Sign/generatedimagestrain"

process_images(image_dir, label_dir, output_dir)
