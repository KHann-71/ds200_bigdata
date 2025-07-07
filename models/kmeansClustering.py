from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pyspark.sql import DataFrame
from torchvision.utils import make_grid
from torch import tensor

class Kmeans:
    def __init__(self, n_clusters=10):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=512)

    def train(self, df: DataFrame) -> List:
        X, y = self._extract_features(df)
        self.model.partial_fit(X)
        return self._evaluate(X, y)

    def predict(self, df: DataFrame) -> List:
        X, y = self._extract_features(df)
        return self._evaluate(X, y)

    def _extract_features(self, df: DataFrame):
        X = np.array(df.select("image").rdd.map(lambda row: row.image).collect()).reshape(-1, 3072)
        y = np.array(df.select("label").rdd.map(lambda row: row.label).collect()).reshape(-1)
        return X, y

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> List:
        cluster_labels = self.model.predict(X)
        reference = self._build_reference(cluster_labels, y)
        pred_labels = np.vectorize(reference.get)(cluster_labels)

        acc = accuracy_score(y, pred_labels)
        prec = precision_score(y, pred_labels, average='macro')
        rec = recall_score(y, pred_labels, average='macro')
        f1 = f1_score(y, pred_labels, average='macro')
        cm = confusion_matrix(y, pred_labels)

        return [pred_labels, acc, self.model.inertia_, prec, rec, f1, cm]

    def _build_reference(self, cluster_labels, y_true):
        ref = {}
        for cluster in np.unique(cluster_labels):
            indices = np.where(cluster_labels == cluster)
            majority_label = np.bincount(y_true[indices]).argmax()
            ref[cluster] = majority_label
        return ref

    def visualize_clusters(self, images: np.ndarray, labels: np.ndarray):
        for c in np.unique(labels):
            imgs = images[labels == c]
            grid = make_grid(tensor(imgs[:25].reshape(-1, 3, 32, 32)), nrow=5)
            plt.figure(figsize=(5, 5))
            plt.imshow(grid.permute(1, 2, 0))
            plt.axis('off')
            plt.title(f"Cluster {c}")
            plt.savefig(f"images/cluster_{c}.png")
