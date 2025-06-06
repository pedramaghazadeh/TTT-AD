import os
import requests
import zipfile
import cv2
import json
from tqdm import tqdm

# Constants
BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
IMAGE_ZIP = "data_object_image_2.zip"
LABEL_ZIP = "data_object_label_2.zip"
OUTPUT_DIR = "/scr/Pedram/VisualLearning/kitti_2d_objects"
CROPPED_DIR = os.path.join(OUTPUT_DIR, "cropped/")
LABEL_JSON = os.path.join(OUTPUT_DIR, "labels.json")

# Download file
def download_file(url, dest):
    if os.path.exists(dest):
        print(f"{dest} already exists, skipping download.")
        return
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    with open(dest, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size=8192)):
            if chunk:
                f.write(chunk)

# Extract zip
def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

# Parse label file
def parse_label_line(line):
    parts = line.strip().split()
    cls = parts[0]
    if cls == "DontCare":
        return None
    xmin, ymin, xmax, ymax = map(int, map(float, parts[4:8]))
    return cls, (xmin, ymin, xmax, ymax)

# Main processing
def process_kitti_2d():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CROPPED_DIR, exist_ok=True)

    # # Step 1: Download
    # download_file(BASE_URL + IMAGE_ZIP, IMAGE_ZIP)
    # download_file(BASE_URL + LABEL_ZIP, LABEL_ZIP)

    # # Step 2: Extract
    extract_zip(IMAGE_ZIP, OUTPUT_DIR)
    extract_zip(LABEL_ZIP, OUTPUT_DIR)

    # Step 3: Parse and crop
    label_dict = {}
    label_dir = os.path.join(OUTPUT_DIR, "training", "label_2")
    image_dir = os.path.join(OUTPUT_DIR, "training", "image_2")

    for fname in tqdm(sorted(os.listdir(label_dir))):
        label_path = os.path.join(label_dir, fname)
        image_path = os.path.join(image_dir, fname.replace(".txt", ".png"))

        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        with open(label_path, "r") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines):
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            cls, (xmin, ymin, xmax, ymax) = parsed

            # Crop image
            crop = image[ymin:ymax, xmin:xmax]
            crop_fname = f"{fname.replace('.txt', '')}_{idx}.png"
            crop_path = os.path.join(CROPPED_DIR, crop_fname)
            cv2.imwrite(crop_path, crop)

            # Save label
            label_dict[crop_fname] = cls

    # Step 4: Save label JSON
    with open(LABEL_JSON, "w") as f:
        json.dump(label_dict, f, indent=2)
    print(f"Saved labels to {LABEL_JSON}")

if __name__ == "__main__":
    process_kitti_2d()