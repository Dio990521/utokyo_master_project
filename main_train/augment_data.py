# main_train/augment_data.py

import os
from PIL import Image
import torch
from torchvision import transforms
import hashlib
import numpy as np

seen_hashes = set()

def image_hash(img: Image.Image):
    arr = np.array(img)
    return hashlib.md5(arr.tobytes()).hexdigest()

def create_augmented_validation_set(
    train_dir,
    val_dir,
    num_augmentations_per_image=25,
    max_trials=500
):
    augmentation_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.15, 0.15),
            fill=255
        )
    ])

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    source_images = [
        f for f in os.listdir(train_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for filename in source_images:
        file_path = os.path.join(train_dir, filename)

        with Image.open(file_path) as img:
            img = img.convert('L')

            seen_hashes = set()
            generated = 0
            trials = 0

            while generated < num_augmentations_per_image and trials < max_trials:
                trials += 1
                aug = augmentation_transform(img)
                h = image_hash(aug)

                if h in seen_hashes:
                    continue

                seen_hashes.add(h)

                base, ext = os.path.splitext(filename)
                save_path = os.path.join(
                    val_dir, f"{base}_aug_{generated}{ext}"
                )
                aug.save(save_path)
                generated += 1

            print(f"{filename}: generated {generated} unique augmentations")

if __name__ == '__main__':
    TRAIN_SKETCHES_DIR = "../data/32x32_final_sketches_test/"
    VALIDATION_SKETCHES_DIR = "../data/32x32_final_sketches_train/"

    VERSIONS_PER_IMAGE = 25

    create_augmented_validation_set(TRAIN_SKETCHES_DIR, VALIDATION_SKETCHES_DIR, VERSIONS_PER_IMAGE)