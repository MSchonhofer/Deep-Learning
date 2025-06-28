import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import albumentations as A
import os
import random
from pathlib import Path
import glob


class TextureDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0

        if self.transform:
            image = self.transform(image=image)['image']

        image = torch.tensor(image)
        if image.ndim == 2:
            image = image.unsqueeze(0) # add one channel (C, H, W) = (1, 128, 128)

        return image, image

def apply_augmentations(patch_size=128, enable_rotation=True):
    transforms_list = [
        A.RandomCrop(width=patch_size, height=patch_size), # patch 128x128
        A.HorizontalFlip(p=0.5), # random horizontal flip
        A.VerticalFlip(p=0.5), # random vertical flip
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5), # random brightness and contrast
        A.RandomGamma(gamma_limit=(80, 120), p=0.5), # random gamma
        A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5 if enable_rotation else 0.0), # random rotate max 10
        # degrees, no zero-valued
    ]
    return A.Compose(transforms_list)


def load_images_from_directory(image_dir, supported_extensions=None):
    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png']
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise ValueError(f"Image directory does not exist: {image_dir}")
    image_paths = []
    for ext in supported_extensions:
        image_paths.extend(glob.glob(str(image_dir / f"*{ext}")))
        image_paths.extend(glob.glob(str(image_dir / f"*{ext.upper()}")))
    image_paths = sorted(list(set(image_paths)))
    print(f"Found {len(image_paths)} images in {image_dir}")
    return image_paths


def generate_patches(images, output_dir='texture_patches', patch_size=128,
                     num_patches=400, enable_rotation=True):
    os.makedirs(f"{output_dir}/train", exist_ok=True)
    os.makedirs(f"{output_dir}/test", exist_ok=True)
    transform = apply_augmentations(patch_size, enable_rotation)
    train_count = int(num_patches * 0.8)
    patch_paths = []
    print(f"Generating {num_patches} patches from {len(images)} source images...")
    for i in range(num_patches):
        img = random.choice(images)
        try:
            augmented = transform(image=img)
            patch = augmented['image']
            split = 'train' if i < train_count else 'test'  # 320 train images, 80 test images
            patch_path = f"{output_dir}/{split}/patch_{i:04d}.jpg"
            cv2.imwrite(patch_path, (patch * 255).astype(np.uint8))
            patch_paths.append(patch_path)
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{num_patches} patches")
        except Exception as e:
            print(f"Error generating patch {i}: {e}")
    print(f"Successfully generated {len(patch_paths)} patches")
    print(f"Training patches: {train_count}")
    print(f"Test patches: {num_patches - train_count}")
    return patch_paths


def preprocess_images_from_directory(image_dir, output_dir='texture_patches',
                                     patch_size=128, num_patches=400, enable_rotation=True):
    image_paths = load_images_from_directory(image_dir)
    valid_images = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.shape[0] >= patch_size and img.shape[1] >= patch_size:
            valid_images.append(img)
            print(f"Loaded: {Path(img_path).name} - Shape: {img.shape}")
        else:
            print(f"Skipped (too small or failed to load): {Path(img_path).name}")
    if not valid_images:
        raise ValueError("No valid images found for processing.")
    return generate_patches(valid_images, output_dir, patch_size, num_patches, enable_rotation)


def load_texture_tiles(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    train_paths, test_paths = [], []
    for ext in ['.jpg', '.jpeg', '.png']:
        train_paths.extend(glob.glob(os.path.join(train_dir, f"*{ext}")))
        train_paths.extend(glob.glob(os.path.join(train_dir, f"*{ext.upper()}")))
        test_paths.extend(glob.glob(os.path.join(test_dir, f"*{ext}")))
        test_paths.extend(glob.glob(os.path.join(test_dir, f"*{ext.upper()}")))
    train_paths = sorted(list(set(train_paths)))
    test_paths = sorted(list(set(test_paths)))
    print(f"Loaded {len(train_paths)} training images and {len(test_paths)} test images")
    return train_paths, test_paths


if __name__ == "__main__":
    print("Starting processing...")
    image_dir = "images"
    output_dir = "texture_patches"
    try:
        patch_paths = preprocess_images_from_directory(
            image_dir=image_dir,
            output_dir=output_dir,
            num_patches=400,
            enable_rotation=True
        )
    except Exception as e:
        print(f"Error processing images from '{image_dir}': {e}")

    train_paths, test_paths = load_texture_tiles(output_dir)
    train_dataset = TextureDataset(train_paths)
    test_dataset = TextureDataset(test_paths)

    print(f"Created datasets - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    if len(train_dataset) > 0:
        sample_image, sample_target = train_dataset[0]
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample target shape: {sample_target.shape}")
        print("Data processing completed successfully!")
