import sys

import numpy as np
from sklearn.base import ClusterMixin, BaseEstimator


class SelfOrganizingMap(BaseEstimator, ClusterMixin):
    """
    This class implements the Self-Organizing-Map algorithm

    Attributes
    ----------
    learning_rate : float
        describes how much new data will change the neurons of this map
    max_iter : int
        sets a maximum number of iterations
    neuron_map_size : int
        sets the number of neurons in the map
    data_dimensions : int
        sets the number of dimensions (features) that datapoints will have
    data_min : float
        the minimum value of all data points
    data_max : float
        the maximum value of all data points
    neuron_map : np.ndarray
        the map of neurons that will be trained
    """

    def __init__(self, *, learning_rate: float = 0.6, max_iter: int = 1000, data_dimensions: int, data_min: float = 0,
                 data_max: float,
                 neuron_map_size: int = 10):
        self.learning_rate: float = learning_rate
        self.max_iter: int = max_iter
        self.neuron_map_size: int = neuron_map_size
        self.data_dimensions: int = data_dimensions
        self.data_min: float = data_min
        self.data_max: float = data_max
        self.neuron_map: np.ndarray = self.init_neurons()

    def init_neurons(self) -> np.ndarray:
        """
        :return: an initialized neuron map with equidistant neurons for this SOM
        """
        linspaces = []
        for i in range(self.data_dimensions):
            linspaces.append(np.linspace(self.data_min, self.data_max, self.neuron_map_size + 1))

        something = np.meshgrid(*linspaces)
        return np.array([subgrid.flatten() for subgrid in something]).T

    def fit(self, X, y=None):
        """
        Fit the SOM
        :param X: an array of datapoints
        :param y: is ignored (dunno why see :func: '~ClusterMixin.fit_predict')
        """
        # TODO implement the cluster algorithm here
        for vector in X:
            pass
        pass

    def find_bmu(self, vector: np.ndarray):
        min_distance = sys.float_info.max
        bmu = None
        for neuron in self.neuron_map:
            distance = np.linalg.norm(vector - neuron)
            if min_distance > distance:
                bmu = neuron
                min_distance = distance
        return bmu, min_distance

    def update_weights(self, vector: np.ndarray):
        for i in range(len(self.neuron_map)):
            neuron = self.neuron_map[i]
            distance = np.linalg.norm(vector - neuron)
            pull_strength = np.interp(distance, [0, self.data_min + self.data_max], [1, 0])
            weights = neuron + self.learning_rate * pull_strength * (vector - neuron)
            self.neuron_map[i] = weights
        pass
 

def main():
    som = SelfOrganizingMap(data_dimensions=3, data_max=10)
    print(som.neuron_map)


if __name__ == '__main__':
    main()
