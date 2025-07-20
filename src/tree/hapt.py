import os
import re
import torch
import numpy as np
from pathlib import Path
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
    def __init__(self, n_clusters, is_leaf, parent=None, points=None, root=None, start_idx=None, end_idx=None, centroid=None):
        """
        n_clusters  : k
        is_leaf     : denotes if tree is leaf node
        parent      : parent HAPT object
        points      : numpy array of points for node
        """
        self.root = root
        if self.root is None:
            self.root = self

        self.is_leaf:bool        = is_leaf
        self.parent:HAPT         = parent
        self.n_clusters:int      = n_clusters
        self.centroid            = centroid
        self.points:np.ndarray   = points

        self.start_idx = start_idx
        self.end_idx = end_idx

        if self.points is not None:
            self.make_points_info()
            self.start_idx = 0
            self.end_idx = len(self.points)
            self.centroid = self.points.mean(axis=0)

        self.cumulated_projection:np.ndarray = None
        self.projection:np.ndarray           = None

        self.branches:list[HAPT] = None if self.is_leaf else [None for i in range(self.n_clusters)]

        super().__init__(n_clusters=n_clusters, **DEFAULT_KMEANS_KWARGS)

    def make_points_info(self):
        self.start_idx = 0
        self.end_idx   = len(self.points)
        self.centroid  = self.points.mean(axis=0)

    def get_points(self):
        """Get points from root based on limit `idx`s"""
        return self.root.points[self.start_idx:self.end_idx]
    
    def get_projected_points(self):
        """Get points projected into cluster's subspace"""
        return self.get_points() @ self.cumulated_projection
    
    def get_projected_centroid(self):
        """Get cluster centroid projected into cluster's subspace"""
        return self.centroid @ self.cumulated_projection
    
    def sort_points(self, start, end, order):
        """Sort portion of `root`'s points (from `start` to `end`), in `order`"""
        self.root.points[start:end] = self.root.points[start:end][order]
        
    def cluster_fit(self):
        """
        Fit k-means cluster on points, extract clusters
        """
        
        # Get K-means labels
        labels = self.fit_predict(self.get_points())

        # Sort points by clusters
        sort_idx = np.argsort(labels)
        counts = np.bincount(labels, minlength=self.n_clusters)
        cutoffs = np.concat(([0, ], np.cumsum(counts))) + self.start_idx
        self.root.sort_points(self.start_idx, self.end_idx, sort_idx)

        # Get cutoff indices (points[start:end]) for each cluster
        idx = [(cutoffs[j], cutoffs[j + 1]) for j in range(self.n_clusters)]
        
        return idx
    
    def branch(self, depth_remaining):
        """
        Recursively generate clusters
        """
        idx = self.cluster_fit()
        centroids = self.cluster_centers_

        is_leaf = (depth_remaining == 1)

        for i in range(self.n_clusters):
            start, end = idx[i]
            if start == end: continue

            self.branches[i] = HAPT(self.n_clusters, is_leaf, parent=self, root=self.root, start_idx=start, end_idx=end, centroid=centroids[i])

            # Recurse non-leaves
            if not is_leaf:
                self.branches[i].branch(depth_remaining - 1)

    def get_projections(self, sizes):
        """
        Get low-rank PCA projections for each local cluster's points. Accumulate
        projection matrices
        """

        # Extract and project points for use
        points_use = self.get_points().copy()
        if self.parent is not None:
            points_use = points_use @ self.parent.cumulated_projection

        # Get properly-sized projection
        #   Note: this may result in nodes with lower-dimension vectors
        n, Din = points_use.shape
        q = min(sizes[0], Din, n - 1)
        if q < 1:
            self.projection = np.eye(Din, dtype=np.float32)
        else:
            _, _, V = pca_lowrank(torch.from_numpy(points_use), q=q)
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

    def save_vectors(self, base_dir):
        """Save projected points and centroid to individual `.npy` files"""
        
        if not self.is_leaf:
            Path(base_dir).mkdir(parents=True, exist_ok=True)
        # Save
        if self.is_leaf:
            np.save(os.path.join(base_dir, "points.npy"), self.get_projected_points())
        try:
            np.save(os.path.join(base_dir, "centroid.npy"), self.get_projected_centroid())
            np.save(os.path.join(base_dir, "projection.npy"), self.projection)
        except AttributeError:
            return
        
        if self.is_leaf:
            return

        # Recurse
        for i in range(self.n_clusters):
            _dir = os.path.join(base_dir, f"{i}")
            if not os.path.exists(_dir):
                os.mkdir(_dir)
            self.branches[i].save_vectors(_dir)

    def save_compressed(self, data, file_path):
        """Save projected vectors and centroids in compressed `.npz`"""

        # Record
        if self.is_leaf:
            data["points"] = self.get_projected_points()
        try:
            data["centroid"] = self.get_projected_centroid()
            data["projection"] = self.projection
        except AttributeError:
            return 
        
        if self.is_leaf:
            return

        # Recurse
        for i in range(self.n_clusters):
            data[i] = {}
            self.branches[i].save_compressed(data[i], file_path)

        if self.parent is None:
            np.savez_compressed(file_path, data=data, allow_pickle=True)

    def load_batches(self, batch_dir):
        """
        Load batches from batch directory, which is a directory full of `batch_<n>.npy` files
        """
        pattern = "batch_([0-9]+)\.npy"

        # Ensure provided path
        if not os.path.exists(batch_dir):
            raise ValueError(f"Batch directory ({batch_dir}) does not exist")
        
        # Get and validate batches
        files = os.listdir(batch_dir)
        files = list(filter(lambda x : re.match(pattern, x), files))
        max_batch = max([int(re.match(pattern, f)[1]) for f in files])

        if len(files) == 0:
            raise ValueError(f"No batches found in {batch_dir}")
        if len(files) != (max_batch + 1):
            print(f"WARNING: You may be missing batches)")

        full_paths = [os.path.join(batch_dir, file) for file in files]

        self.points = np.load(full_paths[0])

        for i in range(1, len(full_paths)):
            path = full_paths[i]
            
            try:
                self.points = np.concat((self.points, np.load(path)), axis=0)
            except ValueError as e:
                if e.args[0].startswith("all the input array dimensions except for the concatenation axis must match exactly"):
                    raise ValueError("All batches must have same dimension")

        self.make_points_info()
        


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


