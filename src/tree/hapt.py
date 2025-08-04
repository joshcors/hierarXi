import os
import re
import json
import tqdm
import torch
import numpy as np
import torch.nn as nn
from uuid import uuid4
from pathlib import Path
from hashlib import sha256
from torch import pca_lowrank
from utils import find_project_root
from sklearn.cluster import MiniBatchKMeans

from utils import PointsManager
from tree.payload_manager import PayloadManger

PACKAGE_DIR = find_project_root()
BASE_DATA_DIR = os.path.join(PACKAGE_DIR, "data", "store")

DEFAULT_KMEANS_KWARGS = {
    "init": "k-means++",
    "max_iter": 300,
    "tol": 0.0001,
    "verbose": 0,
    "random_state": None,
    "copy_x": False,
    "algorithm": "lloyd"
}

class HAPT:
    """
    Hierarchical Adaptive-Projection Tree (HAPT)

    Hierarchical K-means tree, with a unique PCA projection on each graph edge
    """
    def __init__(self, n_clusters, is_leaf, parent=None, points=None, root=None, start_idx=None, end_idx=None, centroid=None, payload_manager=None):
        """
        n_clusters  : k
        is_leaf     : denotes if tree is leaf node
        parent      : parent HAPT object
        points      : numpy array of points for node
        """
        self.root = root
        if self.root is None:
            self.root = self

        self.id = uuid4()
        self.data_dir = self.get_data_dir()

        self.mbk = MiniBatchKMeans(n_clusters=n_clusters,
                                   batch_size=10_000,
                                   max_iter=10,
                                   compute_labels=True,
                                   reassignment_ratio=0,
                                   random_state=0)

        self.is_leaf:bool        = is_leaf
        self.parent:HAPT         = parent
        self.n_clusters:int      = n_clusters
        self.centroid:np.ndarray = centroid
        self.points:np.ndarray   = points

        self.payload_manager:PayloadManger = payload_manager

        self.start_idx = start_idx
        self.end_idx = end_idx

        if self.points is not None:
            self.make_points_info()
            self.start_idx = 0
            self.end_idx = len(self.points)
            self.centroid = self.points.mean(axis=0)
            self.order = np.arange(len(self.points))

        self.cumulated_projection:np.ndarray = None
        self.projection:np.ndarray           = None

        self.branches:list[HAPT] = None if self.is_leaf else [None for i in range(self.n_clusters)]

    def get_data_dir(self):
        return os.path.join(BASE_DATA_DIR, str(self.id))
    
    def get_points_path(self):
        return os.path.join(self.get_data_dir(), "points.npy")
    
    def get_order_path(self):
        return os.path.join(self.get_data_dir(), "order.dat")
    
    def get_info_path(self):
        return os.path.join(self.get_data_dir(), "info.json")

    def make_points_info(self):
        self.start_idx = 0
        self.end_idx   = self.points_manager.shape[0]

        # Root is going to be diverse and large (0 vector good enough)
        self.centroid  = np.zeros(self.points_manager.shape[0], dtype=self.points_manager.dtype)
        self.order = np.arange(len(self.points_manager))

    def get_payload(self, index):
        sort_index = self.root.order[index]
        payload = self.payload_manager.get_payload(sort_index)
        return payload

    def get_points(self):
        """Get points from root based on limit `idx`s"""
        return self.root.points_manager.get_points(self.start_idx, self.end_idx)
    
    def get_projected_points(self):
        """Get points projected into cluster's subspace"""
        return self.get_points() @ self.cumulated_projection
    
    def get_projected_centroid(self):
        """Get cluster centroid projected into cluster's subspace"""
        return self.centroid @ self.cumulated_projection
    
    def sort_points(self, start, end, order):
        """Sort portion of `root`'s points (from `start` to `end`), in `order`
        
        Cyclic reordering to limit to-mem
        """
        order_map = np.memmap(self.get_order_path(), mode="r+", shape=(self.root.points_manager.shape[0], ), dtype=np.int32)
        order_map[start:end] = order_map[order]
        order_map.flush()
        del order_map

        self.root.points_manager.reorder(start, end, order)

    def get_best_leaf(self, vec):
        """
        Naive best vectors retrieval
        """
        if self.is_leaf:
            return self
        
        best = -1
        best_score = -np.inf

        for i, branch in enumerate(self.branches):
            score = branch.centroid.dot(vec)

            if score > best_score:
                best_score = score
                best = i

        return self.branches[best].get_best_leaf(vec)
        
    def cluster_fit(self, **kwargs):
        """
        Fit k-means cluster on points, extract clusters
        """
        
        # Get K-means labels
        if kwargs.get("verbose", False): print("Clustering")
        labels = self.mbk.fit_predict(self.get_points())

        # Sort points by clusters
        sort_idx = np.argsort(labels)
        counts = np.bincount(labels, minlength=self.n_clusters)
        cutoffs = np.concat(([0, ], np.cumsum(counts))) + self.start_idx

        # Get cutoff indices (points[start:end]) for each cluster
        idx = [(cutoffs[j], cutoffs[j + 1]) for j in range(self.n_clusters)]

        if kwargs.get("verbose", False): print("Sorting")
        self.root.sort_points(self.start_idx, self.end_idx, sort_idx + self.start_idx)
        
        return idx
    
    def branch(self, depth_remaining, **kwargs):
        """
        Recursively generate clusters
        """
        if kwargs.get("verbose", False): print(f"Branch {depth_remaining - 1} from leaf")

        idx = self.cluster_fit(**kwargs)
        centroids = self.mbk.cluster_centers_

        is_leaf = (depth_remaining == 1)

        for i in range(self.n_clusters):
            start, end = idx[i]
            if start == end: continue

            if kwargs.get("verbose", False): print(f"Creating branch {i}")
            self.branches[i] = HAPT(self.n_clusters, is_leaf, parent=self, root=self.root, start_idx=start, end_idx=end, centroid=centroids[i])

            # Recurse non-leaves
            if not is_leaf:
                self.branches[i].branch(depth_remaining - 1, **kwargs)

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

    def load_batches(self, batch_dir, batch_size=1000, dim=1024):
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
        files = sorted(files, key=lambda x : int(re.match(pattern, x)[1]))
        max_batch = int(re.match(pattern, files[-1])[1])

        if len(files) == 0:
            raise ValueError(f"No batches found in {batch_dir}")
        if len(files) != (max_batch + 1):
            print(f"WARNING: You may be missing batches)")

        full_paths = [os.path.join(batch_dir, file) for file in files]
        full_paths = [os.path.abspath(path) for path in full_paths]

        self.id = sha256("".join(full_paths).encode("utf-8")).hexdigest()
        self.data_dir = self.get_data_dir()
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        N = len(full_paths) * batch_size

        points_path = self.get_points_path()
        info_path = self.get_info_path()

        if not Path(points_path).exists() or not Path(info_path).exists():
            points_memmap = np.memmap(points_path, mode="w+", dtype=np.float16, shape=(N, dim))

            cursor = 0
            for path in tqdm.tqdm(full_paths):

                batch = np.load(path, mmap_mode="r")
                n = batch.shape[0]
                points_memmap[cursor:cursor+n] = batch
                cursor += n

            points_memmap.flush()

            with open(info_path, "w") as f:
                json.dump({"cursor": cursor}, f)

        # Recover cursor
        if Path(info_path).exists():
            with open(info_path, "r") as f:
                info = json.load(f)
            cursor = info["cursor"]

        order_path = self.get_order_path()
        if not Path(order_path).exists():
            order = np.arange(cursor).astype(np.int32)
            order_memmap = np.memmap(order_path, mode="w+", shape=(cursor, ), dtype=np.int32)
            order_memmap[:] = order
            order_memmap.flush()
            del order_memmap

        self.points_manager = PointsManager(points_path, (cursor,dim), dtype=np.float16)
        self.make_points_info()
        self.payload_manager.load_info_and_order(self.data_dir)
        


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


