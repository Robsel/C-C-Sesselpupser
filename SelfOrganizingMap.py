import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator


class SelfOrganizingMap(BaseEstimator, ClusterMixin):
    def __init__(self, *, learning_rate=0.6, max_iter=1000, data_dimensions=2, neuron_map_size=10):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.neuron_map_size = neuron_map_size
        self.data_dimensions = data_dimensions
        self.neuron_map = self.init_neurons()
        pass

    def init_neurons(self):
        linspaces = []
        for i in range(self.data_dimensions):
            linspaces.append(np.linspace(0, self.neuron_map_size, self.neuron_map_size + 1))

        something = np.meshgrid(*linspaces)
        return np.array([subgrid.flatten() for subgrid in something]).T

    def fit(self, X, y=None):
        # TODO implement the cluster algorithm here
        pass


def main():
    som = SelfOrganizingMap(data_dimensions=3)
    print(som.neuron_map)


if __name__ == '__main__':
    main()
