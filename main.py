from scripts.data_preprocessing import load_reference_image, load_current_images
from scripts.model_training import train_model
from scripts.image_comparison import compare_images
import numpy as np
import tensorflow as tf
import cv2
import os
import json
from typing import List, Tuple, Dict
import warnings
from tensorflow.keras.mixed_precision import set_global_policy

def main():
    # Enable mixed precision training for better performance
    set_global_policy('mixed_float16')

    # Suppress warnings
    warnings.filterwarnings('ignore')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Configure memory growth and optimization
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Use only one GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
        except RuntimeError as e:
            print(e)

    # Set up TF configuration
    tf.config.optimizer.set_jit(True)  # Enable XLA optimization
    tf.data.experimental.enable_debug_mode()
    # Load and preprocess images
    reference_image = load_reference_image('data/reference_images/1.jpg')
    current_images = load_current_images('data/current_images/')

    # Train model with optimized parameters
    model = train_model(reference_image)

    # Process each current image
    for i, current_image in enumerate(current_images):
        # Compare images and get similarity map and anomalies
        similarity_map, anomalies = compare_images(
            model,
            reference_image,
            current_image
        )

        # Generate and save report
        report = generate_report(f'image_{i+1}.jpg', anomalies, similarity_map)
        with open(f'output/report_{i+1}.json', 'w') as f:
            json.dump(report, f, indent=4)

        # Visualize and save results
        visualize_results(
            current_image,
            anomalies,
            f'output/result_{i+1}.png'
        )

        print(f"Processed image {i+1}")
        print(f"Found {len(anomalies)} anomalies")
        print("-" * 50)

if __name__ == "__main__":
    main()