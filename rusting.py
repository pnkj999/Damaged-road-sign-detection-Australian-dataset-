import os
import cv2
import numpy as np
from noise import pnoise2

# === Perlin Noise Generator ===
def generate_perlin_noise(width, height, scale=150, octaves=6, persistence=0.5, lacunarity=2.0, seed=None):
    noise_img = np.zeros((height, width), dtype=np.float32)
    seed = np.random.randint(0, 100) if seed is None else seed

    for y in range(height):
        for x in range(width):
            noise_val = pnoise2(x / scale,
                                y / scale,
                                octaves=octaves,
                                persistence=persistence,
                                lacunarity=lacunarity,
                                repeatx=width,
                                repeaty=height,
                                base=seed)
            noise_img[y][x] = (noise_val + 1) / 2
    return noise_img

# === Rust Texture Creator ===
def create_rust_texture(noise_map):
    rust_texture = np.zeros((noise_map.shape[0], noise_map.shape[1], 3), dtype=np.uint8)
    for y in range(noise_map.shape[0]):
        for x in range(noise_map.shape[1]):
            val = noise_map[y, x]
            r = int(70 + val * 150)
            g = int(50 + val * 80)
            b = int(40 + val * 60)
            rust_texture[y, x] = (b, g, r)
    
    
        # Convert to HSV, reduce saturation
    hsv = cv2.cvtColor(rust_texture, cv2.COLOR_BGR2HSV)
    hsv[...,1] = hsv[...,1] * 0.6  # reduce saturation (0.5–0.7 for dullness)
    rust_texture = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return rust_texture

# === Blender ===
def blend_with_image(base_img, rust_texture, alpha_map, intensity=2.0):
    alpha_map = cv2.normalize(alpha_map, None, 0, 1, cv2.NORM_MINMAX)
    alpha_map = cv2.GaussianBlur(alpha_map, (9, 9), 0)
    alpha_map = np.clip(alpha_map * intensity, 0, 1)
    alpha_map = np.expand_dims(alpha_map, axis=2)

    rust_texture1 = rust_texture.astype(np.float32)
    rust_texture = cv2.GaussianBlur(rust_texture1, (5, 5), 0)

    base_img = base_img.astype(np.float32)

    blended = alpha_map * rust_texture + (1 - alpha_map) * base_img
    return np.clip(blended, 0, 255).astype(np.uint8)

# === Process Image ===
def apply_rust_effect_to_image(base_img):
    height, width = base_img.shape[:2]
    noise1 = generate_perlin_noise(width, height, scale=60)
    noise2 = generate_perlin_noise(width, height, scale=90)
    noise3 = generate_perlin_noise(width, height, scale=120)
    combined_noise = (noise1 + noise2 + noise3) / 3.0
    combined_noise = np.clip(combined_noise ** 1.2, 0, 1)

    rust_texture = create_rust_texture(combined_noise)
    return blend_with_image(base_img, rust_texture, combined_noise, intensity=1.0)

# === Batch Process Directory ===
def apply_rust_to_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    for filename in os.listdir(input_dir):
        if any(filename.lower().endswith(ext) for ext in image_extensions):
            img_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            print(f"Processing: {filename}")
            image = cv2.imread(img_path)
            if image is None:
                print(f"❌ Failed to load {filename}")
                continue

            rusty_image = apply_rust_effect_to_image(image)
            cv2.imwrite(output_path, rusty_image)

    print("✅ Rusting completed for all images.")

# === RUN ===
input_folder = "/home/msi/course/ACS/croppedimages/geneatedimages"
output_folder = "/home/msi/course/ACS/croppedimages/outputrustingeffect"
apply_rust_to_directory(input_folder, output_folder)

