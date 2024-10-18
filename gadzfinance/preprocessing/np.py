from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


def pca(data: np.ndarray, n_components: float | int) -> np.ndarray:
    data = data.copy()
    scaler = StandardScaler()
    model = PCA(n_components=n_components)
    data = scaler.fit_transform(data)
    data = model.fit_transform(data)
    data = model.inverse_transform(data)
    data = scaler.inverse_transform(data)
    return data
