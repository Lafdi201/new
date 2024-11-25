import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from scipy.spatial import distance
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
import numpy as np

def calculate_pixels_per_cm(reference_width_pixels: int, actual_width_cm: float) -> float:
    return reference_width_pixels / actual_width_cm

def load_reference_image(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} not found or is corrupt.")
    image = cv2.resize(image, target_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_float = img_to_array(image) / 255.0
    return image_float  # Return just the processed image

def load_current_images(image_dir: str, target_size: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
    images = []
    for image_name in sorted(os.listdir(image_dir)):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_dir, image_name)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Image {image_name} is corrupt or unreadable. Skipping.")
                continue
            image = cv2.resize(image, target_size)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_float = img_to_array(image_rgb) / 255.0
            images.append(image_float)
    return np.array(images)