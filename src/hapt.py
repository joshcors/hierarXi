import os
import torch
import numpy as np
from torch import pca_lowrank
from sklearn.cluster import KMeans

import torch.nn as nn

DEFAULT_KMEANS_KWARGS = {
    "init": "k-means++",
    "max_iter": 300,
    "tol": 0.0001,
    "verbose": 0,
    "random_state": None,
    "copy_x": False,
    "algorithm": "lloyd"
}

class HAPT(KMeans):
    """
    Hierarchical Adaptive-Projection Tree (HAPT)

    Hierarchical K-means tree, with a unique PCA projection on each graph edge
    """
    def __init__(self, n_clusters, is_leaf, parent, points, base_dir=None):
        """
        n_clusters  : k
        is_leaf     : denotes if tree is leaf node
        parent      : parent HAPT object
        points      : numpy array of points for node
        """
        self.is_leaf:bool        = is_leaf
        self.parent:HAPT         = parent
        self.n_clusters:int      = n_clusters
        self.points:np.ndarray   = points
        self.centroid:np.ndarray = self.points.mean(axis=0)
        self.base_dir:str        = base_dir

        self.cumulated_projection:np.ndarray = None
        self.projection:np.ndarray           = None

        self.branches:list[HAPT] = None if self.is_leaf else [None for i in range(self.n_clusters)]

        super().__init__(n_clusters=n_clusters, **DEFAULT_KMEANS_KWARGS)

    def cluster_fit(self):
        """
        Fit k-means cluster on points, extract clusters
        """
        
        labels = self.fit_predict(self.points)

        clusters = [
            self.points[labels == i] for i in range(self.n_clusters)
        ]
        
        return clusters
    
    def branch(self, depth_remaining):
        """
        Recursively generate clusters
        """
        clusters = self.cluster_fit()

        is_leaf = (depth_remaining == 1)

        for i in range(self.n_clusters):
            self.branches[i] = HAPT(self.n_clusters, is_leaf, self, clusters[i])

            if not is_leaf:
                self.branches[i].branch(depth_remaining - 1)

    def get_projections(self, sizes):
        """
        Get low-rank PCA projections for each local cluster's points. Accumulate
        projection matrices
        """

        # Extract and project points for use
        points_use = self.points.copy()
        if self.parent is not None:
            points_use = points_use @ self.parent.cumulated_projection

        # Get properly-sized projection
        #   Note: this may result in nodes with lower-dimension vectors
        q = min(sizes[0], points_use.shape[1], points_use.shape[0] - 1)
        _, _, V = pca_lowrank(torch.tensor(points_use), q=q)

        self.projection = V.numpy()

        # Accumulate projections
        if self.parent is None:
            self.cumulated_projection = self.projection
        else:
            self.cumulated_projection = self.parent.cumulated_projection @ self.projection

        if self.is_leaf:
            return
        
        # Recurse
        for i in range(self.n_clusters):
            self.branches[i].get_projections(sizes[1:])

    def save_vectors(self, base_dir=None):
        if base_dir is None:
            base_dir = self.base_dir
        if base_dir is None:
            print("Please provide a directory")
            return
        
        points_use = self.points @ self.cumulated_projection

        np.save(os.path.join(base_dir, "points.npy"), points_use)
        try:
            np.save(os.path.join(base_dir, "centroids.npy"), self.cluster_centers_)
        except AttributeError:
            return

        for i in range(self.n_clusters):
            _dir = os.path.join(base_dir, f"{i}")
            if not os.path.exists(_dir):
                os.mkdir(_dir)
            self.branches[i].save_vectors(_dir)

    def save_compressed(self, data, file_path):
        points_use = self.points @ self.cumulated_projection

        data["points"] = points_use
        try:
            data["centroids"] = self.cluster_centers_
        except AttributeError:
            return 
        
        for i in range(self.n_clusters):
            data[i] = {}
            self.branches[i].save_compressed(data[i], file_path)

        if self.parent is None:
            np.savez_compressed(file_path, data=data, allow_pickle=True)

class HAPTPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)

        middle_dim = int(np.sqrt(input_dim * output_dim))
        middle_dim = max(middle_dim, 64)

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.lin1 = nn.Linear(input_dim, middle_dim)
        self.lin2 = nn.Linear(middle_dim, output_dim)

    def forward(self, x):
        x = nn.GELU(self.lin1(x))
        return self.lin2(x)


