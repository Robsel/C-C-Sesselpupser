import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator


class SelfOrganizingMap(BaseEstimator, ClusterMixin):
    def __init__(self, *, learning_rate=0.6, max_iter=1000, som_dimension=(10, 10)):
        pass

    def fit(self, X, y=None):
        # TODO implement the cluster algorithm here
        pass
