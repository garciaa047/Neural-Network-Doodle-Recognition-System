import os
import numpy as np
import pandas as pd
from PIL import Image, ImageChops
import random
import constants as con

MAX_ROTATION = 15
MAX_SHIFT = 5

# --- Augment image copies ---
def augment_image(img):
    # Random rotation
    angle = random.uniform(-MAX_ROTATION, MAX_ROTATION)
    # Fill corners with white, 255 (Has to be changed if images have black background)
    img = img.rotate(angle, fillcolor=255)

    # Random translation
    dx = random.randint(-MAX_SHIFT, MAX_SHIFT)
    dy = random.randint(-MAX_SHIFT, MAX_SHIFT)
    img = ImageChops.offset(img, dx, dy)

    # Convert to numpy for noise
    arr = np.array(img).astype(np.float32)

    # Add Gaussian noise
    noise_std = random.uniform(0.01, 0.03)
    noise = np.random.normal(0, noise_std * 255, arr.shape)
    arr += noise

    # Clip back to valid range
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


# --- Flatten Image ---
def image_to_row(img, label):
    img = img.resize(con.IMAGE_SIZE)
    # Normalize pixel values
    pixels = np.array(img) / 255.0
    return [label] + pixels.flatten().tolist()



# --- Update constants.py with new classes ---
def update_constants_with_classes(selected_classes):
    const_file = con.FILE_PATH_DIR + "constants.py"

    with open(const_file, "r") as f:
        lines = f.readlines()

    # Replace the constants.py file with the new version that contains updated class names 
    lines = [line for line in lines if not line.strip().startswith("CLASS_NAMES")]
    lines.append(f'CLASS_NAMES = {selected_classes}\n')

    with open(const_file, "w") as f:
        f.writelines(lines)



# --- Main ---
def main():
    # Get list of all classes (Image Types)
    all_classes = [
        d for d in os.listdir(con.DATASET_DIR)
        if os.path.isdir(os.path.join(con.DATASET_DIR, d))
    ]

    # Randomly select NUM_ITEMS number of classes from total
    selected_classes = random.sample(
        all_classes, min(con.NUM_ITEMS, len(all_classes))
    )
    # Change class name to int value
    # Stored later so that int value can be converted back to class name
    class_to_idx = {c: i for i, c in enumerate(selected_classes)}

    rows = []

    for label in selected_classes:
        label_dir = os.path.join(con.DATASET_DIR, label)
        encoded_label = class_to_idx[label]

        # Find all img files
        filenames = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        # Take IMAGES_PER_ITEM number of images randomly from list of images
        filenames = random.sample(
            filenames, min(con.IMAGES_PER_ITEM, len(filenames))
        )


        for filename in filenames:
            path = os.path.join(label_dir, filename)

            # Convert to greyscale
            img = Image.open(path).convert("L")

            # Add original copy
            rows.append(image_to_row(img, encoded_label))

            # Add AUGMENT_COPIES number of augmented copies
            for _ in range(con.AUGMENT_COPIES):
                aug_img = augment_image(img)
                rows.append(image_to_row(aug_img, encoded_label))

    # Column Headers
    columns = ["label"] + [f"pixel{i}" for i in range(con.NUM_PIXELS)]
    df = pd.DataFrame(rows, columns=columns)
    # Save array to CSV
    df.to_csv(con.CSV_FILE, index=False)

    print(f"Saved {len(df)} samples to {con.CSV_FILE}")

    update_constants_with_classes(selected_classes)
    print(f"Updated CLASS_NAMES: {selected_classes}")


if __name__ == "__main__":
    main()
