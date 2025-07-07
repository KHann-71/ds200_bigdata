from typing import List
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.utils import parallel_backend
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pyspark.sql import DataFrame

class SVM:
    def __init__(self, n_components: int = 100):
        self.pca = PCA(n_components=n_components)
        self.model = MLPClassifier(hidden_layer_sizes=(512, 256), activation='relu',
                                   max_iter=1, warm_start=True, random_state=42)

    def train(self, df: DataFrame) -> List:
        X = np.array(df.select("image").rdd.map(lambda row: row.image).collect()).reshape(-1, 3072)
        y = np.array(df.select("label").rdd.map(lambda row: row.label).collect()).reshape(-1)

        X = self.pca.fit_transform(X)
        with parallel_backend("loky", n_jobs=4):
            self.model.fit(X, y)

        pred = self.model.predict(X)
        return self.evaluate(y, pred)

    def predict(self, df: DataFrame) -> List:
        X = np.array(df.select("image").rdd.map(lambda row: row.image).collect()).reshape(-1, 3072)
        y = np.array(df.select("label").rdd.map(lambda row: row.label).collect()).reshape(-1)
        X = self.pca.transform(X)
        pred = self.model.predict(X)
        return self.evaluate(y, pred, return_cm=True)

    def evaluate(self, y_true, y_pred, return_cm=False):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred) if return_cm else None
        return (y_pred, acc, prec, rec, f1) if not return_cm else (y_pred, acc, prec, rec, f1, cm)
