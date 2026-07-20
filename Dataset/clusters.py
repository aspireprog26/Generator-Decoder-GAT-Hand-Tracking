import torch
import joblib
import hdbscan 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from optimizedataset import DatasetOptimizer

class Clusters:
    def __init__(self):
        self.data_dir = Path("/home/mrtcloud-1/Documents/StereoDataset")
        self.cluster_dir = self.data_dir / "Clusters"
        self.types = ["Clean", "Noisy"]

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size = 8,
            min_samples = 2,
            prediction_data = True
        )

    def loadData(self):
        optimizer = DatasetOptimizer()
        data = []

        for type in self.types:
            for pt in (self.data_dir / f'{type}Normalized').glob("*.pt"):
                point = (torch.load(pt)[3]).numpy()
                point_normalized = (optimizer.getFeatures(point, None)[0]).flatten()
                data.append(point_normalized)
        
        data = np.stack(data)
        return data

    def createClusters(self):
        data = self.loadData()
        data = StandardScaler().fit_transform(data)
        self.clusterer.fit(data)

        labels = self.clusterer.labels_
        print(labels.max())
        print(np.count_nonzero(labels == -1))
        data_2d = PCA(n_components = 2).fit_transform(data)
        plt.scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            c = labels, 
            s = 1
        )
        plt.title("Pose Clusters PCA Representation")
        plt.show()

clusters = Clusters()
clusters.createClusters()