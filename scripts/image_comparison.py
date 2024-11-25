import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from datetime import datetime
import json

def compare_images(model, reference_image, test_image, threshold=0.9):
    # Get feature maps from intermediate layer
    feature_model = Model(inputs=model.input[0], outputs=model.get_layer('dense_1').output)
    
    # Initialize similarity map
    similarity_map = np.zeros((128, 128))
    window_size = 32
    stride = 16

    # Sliding window comparison
    for y in range(0, 128 - window_size, stride):
        for x in range(0, 128 - window_size, stride):
            ref_window = reference_image[0, y:y+window_size, x:x+window_size]
            test_window = test_image[0, y:y+window_size, x:x+window_size]
            
            ref_features = feature_model.predict(np.expand_dims(ref_window, 0))
            test_features = feature_model.predict(np.expand_dims(test_window, 0))
            
            similarity = 1 / (1 + np.linalg.norm(ref_features - test_features))
            similarity_map[y:y+window_size, x:x+window_size] = similarity

    # Find regions with low similarity
    dissimilar_regions = np.where(similarity_map < threshold)
    anomalies = []
    
    if len(dissimilar_regions[0]) > 0:
        # Use DBSCAN clustering to group nearby points
        from sklearn.cluster import DBSCAN
        points = np.column_stack(dissimilar_regions)
        clustering = DBSCAN(eps=10, min_samples=5).fit(points)
        
        # Get unique clusters
        unique_labels = np.unique(clustering.labels_)
        
        # For each cluster, calculate center and radius
        for label in unique_labels:
            if label == -1:  # Skip noise
                continue
            cluster_points = points[clustering.labels_ == label]
            center = np.mean(cluster_points, axis=0)
            radius = np.max(np.linalg.norm(cluster_points - center, axis=1))
            anomalies.append((int(center[1]), int(center[0]), int(radius)))

    return similarity_map, anomalies

def generate_report(image_path, anomalies, similarity_map):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report = {
        "timestamp": timestamp,
        "image_path": image_path,
        "total_components_checked": int(similarity_map.size),
        "anomalies_detected": len(anomalies),
        "average_similarity": float(np.mean(similarity_map)),
        "min_similarity": float(np.min(similarity_map)),
        "max_similarity": float(np.max(similarity_map))
    }
    
    return report

def visualize_results(original_image, anomalies, output_path):
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    for (x, y, r) in anomalies:
        circle = plt.Circle((x, y), r, color='red', fill=False, linewidth=2)
        plt.gca().add_patch(circle)
    
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()