# scripts/precompute_knn.py

import pandas as pd
import numpy as np
from utils.umap_graph import compute_knn_graph
import torch
import os

def precompute_knn(data: pd.DataFrame, feature_dir: str, k: int = 10, output_path: str = "knn_graph.pt"):
    features = []
    for idx in range(len(data)):
        filename = data.iloc[idx]['filename_column'][:-4] + ".pt"
        filepath = os.path.join(feature_dir, filename)
        feature = torch.load(filepath).numpy()
        # Assume _layer_aggregation_strategy is 'mean' as in your Dataset
        feature = feature.mean(axis=1)
        features.append(feature.flatten())

    features = np.array(features)
    edges = compute_knn_graph(features, k=k)
    torch.save(edges, output_path)
    print(f"KNN graph saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    data = pd.read_csv("path_to_your_data.csv")  # Replace with your actual data loading
    precompute_knn(data, feature_dir="path_to_features", k=10, output_path="knn_graph.pt")
