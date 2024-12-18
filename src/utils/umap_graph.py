# utils/umap_graph.py

import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm

def compute_knn_graph(features: np.ndarray, k: int = 10) -> List[Tuple[int, int, float]]:
    """
    Compute the K-Nearest Neighbors graph for the given features.

    Returns a list of tuples (i, j, distance).
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(features)
    distances, indices = nbrs.kneighbors(features)

    edges = []
    for i in tqdm(range(len(features))):
        for j in range(1, k):  # Skip the first neighbor (itself)
            edges.append((i, indices[i][j], distances[i][j]))
    return edges
