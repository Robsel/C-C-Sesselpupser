import numpy as np

class SOM:
    def __init__(self, input_dim, som_dim=(10, 10), learning_rate=0.1, radius=None):
        self.input_dim = input_dim
        self.som_dim = som_dim
        self.learning_rate = learning_rate
        self.radius = radius if radius else max(som_dim) / 2
        self.weights = np.random.random((som_dim[0], som_dim[1], input_dim))
        self.iteration = 0

    def _find_bmu(self, vector):
        distances = np.linalg.norm(self.weights - vector, axis=2)
        return np.unravel_index(distances.argmin(), distances.shape)

    def _update_weights(self, vector, bmu):
        # Calculate the distance from each neuron to the BMU
        d = np.linalg.norm(np.array(list(np.ndindex(self.weights.shape[:2]))) - np.array(bmu), axis=1)
        d = d.reshape(self.som_dim[0], self.som_dim[1])  # Reshape d to match the SOM dimensions

        # Compute the influence of the BMU on each neuron
        influence = np.exp(-(d**2) / (2 * (self.radius**2)))

        # Learning rate adjustment (using broadcasting)
        lr_adjusted = self.learning_rate * influence

        # Update weights
        for i in range(self.som_dim[0]):
            for j in range(self.som_dim[1]):
                self.weights[i, j, :] += lr_adjusted[i, j] * (vector - self.weights[i, j, :])

    def train(self, data, num_iterations):
        for i in range(num_iterations):
            for vector in data:
                bmu = self._find_bmu(vector)
                self._update_weights(vector, bmu)
            self.iteration += 1
            self.radius = max(self.som_dim) / 2 / (1 + i / (num_iterations / 2))

    def map_vects(self, data):
        mapped = np.array([self.weights[self._find_bmu(v)] for v in data])
        return mapped.reshape(len(data), self.input_dim)

# Example Usage
input_data = np.random.rand(100, 5)  # 100 data points, each with 5 dimensions
som = SOM(input_dim=5, som_dim=(10, 10))
som.train(input_data, num_iterations=100)
output_data = som.map_vects(input_data)
