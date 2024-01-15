import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Functions for initializing the grid
def initializegrid_onepoint(num_neurons, data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    mid_points = (min_vals + max_vals) / 2
    return np.tile(mid_points, (num_neurons, 1))

def initializegrid_diagonal(num_neurons, data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    linspace_arrays = [np.linspace(min_vals[i], max_vals[i], num_neurons) for i in range(len(min_vals))]
    return np.column_stack(linspace_arrays)

def initialize_random(num_neurons, data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    return np.random.uniform(min_vals, max_vals, (num_neurons, data.shape[1]))

# SOM class
class SOM:
    def __init__(self, data, num_neurons, epochs, learning_rate, influence, update_neighbors_epoch, calculate_k_epoch, k_neighbors, randomize_data=True, init_mode='diagonal'):
        self.data = np.array(data)
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.influence = influence
        self.update_neighbors_epoch = update_neighbors_epoch
        self.calculate_k_epoch = calculate_k_epoch
        self.k_neighbors = k_neighbors
        self.randomize_data = randomize_data
        self.init_mode = init_mode

        if init_mode == 'diagonal':
            self.weights = initializegrid_diagonal(num_neurons, self.data)
        elif init_mode == 'one_point':
            self.weights = initializegrid_onepoint(num_neurons, self.data)
        elif init_mode == 'random':
            self.weights = initialize_random(num_neurons, self.data)
        else:
            raise ValueError("Invalid init_mode. Choose 'diagonal', 'one_point', or 'random'")
        self.neighbors = {i: [] for i in range(num_neurons)}

    def calculate_k_closest_neighbors(self):
        distances = np.linalg.norm(self.weights[:, None] - self.weights, axis=2)
        for i in range(self.num_neurons):
            self.neighbors[i] = np.argsort(distances[i])[:self.k_neighbors]

    def update_neighborhood(self, neuron_idx, current_data_point, learning_rate, influence):
        for neighbor_idx in self.neighbors[neuron_idx]:
            self.weights[neighbor_idx] += influence * learning_rate * (current_data_point - self.weights[neighbor_idx])

    def train(self, step_by_step=False):
        if step_by_step:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))

        for epoch in range(self.epochs):
            if self.randomize_data:
                np.random.shuffle(self.data)
            for x in self.data:
                closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
                self.weights[closest_neuron_idx] += self.learning_rate * (x - self.weights[closest_neuron_idx])

                if epoch >= self.update_neighbors_epoch and epoch % self.update_neighbors_epoch == 0:
                    self.update_neighborhood(closest_neuron_idx, x, self.learning_rate, self.influence)

            if epoch >= self.calculate_k_epoch and epoch % self.calculate_k_epoch == 0:
                self.calculate_k_closest_neighbors()

            if step_by_step:
                ax.clear()
                ax.scatter(self.data[:, 0], self.data[:, 1], label='Data Points')
                ax.scatter(self.weights[:, 0], self.weights[:, 1], marker='o', color='red', label='Neuron Positions')
                ax.set_title(f'SOM Training Epoch {epoch + 1}')
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.legend()
                plt.draw()
                plt.pause(0.05)

        if step_by_step:
            plt.ioff()

        return self.map_data()

    def map_data(self):
        clusters = []
        for x in self.data:
            closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
            clusters.append(closest_neuron_idx)
        cluster_column = np.array(clusters).reshape(-1, 1)
        return cluster_column

# Example usage
data, _ = make_blobs(n_samples=120, centers=8, n_features=7, random_state=42)
num_neurons = int(np.ceil(np.sqrt(data.shape[0])))
epochs = 10
learning_rate = 0.3
update_neighbors_epoch = 3
influence = 0.1
calculate_k_epoch = 3
k_neighbors = 5

som = SOM(data, num_neurons, epochs, learning_rate, influence, update_neighbors_epoch, calculate_k_epoch, k_neighbors, randomize_data=True, init_mode='random')
print(som.train(step_by_step=True))
