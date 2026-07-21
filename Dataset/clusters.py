import umap
import torch
import joblib
import hdbscan 
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from sklearn.decomposition import PCA
from optimizedataset import DatasetOptimizer
from sklearn.preprocessing import StandardScaler

class Clusters:
    def __init__(self):
        self.data_dir = Path("/home/mrtcloud-1/Documents/StereoDataset")
        self.cluster_dir = self.data_dir / "Clusters"
        self.types = ["Clean", "Noisy"]

        self.labels = None
        self.data = None
        self.errors = None

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size = 50,
            min_samples = 5,
            prediction_data = True
        )

    def loadData(self):
        optimizer = DatasetOptimizer()
        data = []
        errors = []

        for type in self.types:
            for pt in (self.data_dir / f'{type}Normalized').glob("*.pt"):
                point = (torch.load(pt)[2]).numpy()
                point_normalized = (optimizer.getFeatures(point, None)[0]).flatten()
                error = (torch.load(pt)[3]).numpy()
                errors.append(error)
                data.append(point_normalized)
        
        data = np.stack(data)
        errors = np.stack(errors)
        return data, errors

    def saveMeanData(self, labels: np.ndarray, errors: np.ndarray):
        for i in range(-1, labels.max() + 1):
            cluster_path = self.cluster_dir / f'meanmat{i}.npy'
            cluster_points = errors[labels == i]
            mean_matrix = np.mean(cluster_points, dim = 0)
            np.save(mean_matrix, cluster_path)

    def createClusters(self):
        self.data, self.errors = self.loadData()
        data_normalized = StandardScaler().fit_transform(self.data)
        
        data_pca = PCA(n_components = 0.95).fit_transform(data_normalized)         # Keep top k components that describe 95% of the variance
        self.clusterer.fit(data_pca)
        self.labels = self.clusterer.labels_

        print(self.labels.max())
        print(np.count_nonzero(self.labels == -1))

        reducer = umap.UMAP(
            n_components = 2,
            random_state = 42
        )
        data_2d = reducer.fit_transform(data_pca)

        plt.scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            c = self.labels, 
            s = 1
        )
        plt.title("Pose Clusters PCA Representation")
        plt.show()

    def computeCovariances(self):
        self.data = self.data.reshape(self.data.shape[0], 21, 7)
        numpy2ri.activate()
        ro.r('library(mixMatrix)')

        r_data = ro.ListVector({
            str(i + 1): ro.r.matrix(self.data[i, :, :], nrow = 21, ncol = 7)
            for i in range(self.data.shape[0])
        })
        
        fit = ro.r('''
            function (data) {
                fit <- matrixNormal(data)
                return fit
            }

        ''')(r_data)

        row_cov = np.array(fit.rx2("U"))
        col_cov = np.array(fit.rx2("V"))

        chol_row = np.linalg.cholesky(row_cov)
        chol_col = np.linalg.cholesky(col_cov)

        np.save(chol_row, self.cluster_dir / "cholrow.npy")
        np.save(chol_col, self.cluster_dir / "cholcol.npy")

    def getClusterData(self):
        joblib.dump(self.clusterer, self.cluster_dir / "poseclusterer.joblib")
        self.saveMeanData(self.labels, self.errors)
        self.computeCovariances()

clusters = Clusters()
clusters.createClusters()
clusters.getClusterData()