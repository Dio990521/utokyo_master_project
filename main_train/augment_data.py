# main_train/augment_data.py

import os
from PIL import Image
import torch
from torchvision import transforms


def create_augmented_validation_set(train_dir, val_dir, num_augmentations_per_image=1):
    augmentation_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            fill=255
        )
    ])

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)
        print(f"Created directory: {val_dir}")

    source_images = [f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(source_images)} source images in {train_dir}.")

    for filename in source_images:
        file_path = os.path.join(train_dir, filename)

        with Image.open(file_path) as img:
            img = img.convert('L')

            for i in range(num_augmentations_per_image):
                augmented_img = augmentation_transform(img)

                base_name, extension = os.path.splitext(filename)
                new_filename = f"{base_name}_aug_{i}{extension}"
                save_path = os.path.join(val_dir, new_filename)
                augmented_img.save(save_path)
                print(f"  -> Saved augmented image to {save_path}")

    print("\nData augmentation complete!")


if __name__ == '__main__':
    TRAIN_SKETCHES_DIR = "../envs/drawing_env/training/sketches/"
    VALIDATION_SKETCHES_DIR = "../envs/drawing_env/training/sketches/"

    VERSIONS_PER_IMAGE = 7

    create_augmented_validation_set(TRAIN_SKETCHES_DIR, VALIDATION_SKETCHES_DIR, VERSIONS_PER_IMAGE)