import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def initializegrid(num_neurons, data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    grid_dim = int(np.ceil(np.sqrt(num_neurons)))
    grids = [np.linspace(min_vals[i], max_vals[i], grid_dim) for i in range(data.shape[1])]
    neuron_positions = np.array(np.meshgrid(*grids)).T.reshape(-1, data.shape[1])
    return neuron_positions
    
class SOM:
    def __init__(self, data, num_neurons, epochs, learning_rate):
        self.data = np.array(data)
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = initializegrid(num_neurons, self.data) 

    def update_neighborhood(self, neuron_idx, closest_neuron_idx, current_data_point, learning_rate):
        influence = 0.01
        self.weights[neuron_idx-1] += influence * learning_rate * (current_data_point - self.weights[neuron_idx-1])

    def find_neighborhood(self, closest_neuron_idx, current_data_point, learning_rate):
        n = num_neurons
        grid = max(int(np.sqrt(self.data.shape[0])), 4)
        for i in range(n):
            if i > 0 and (i - 1) % grid != 0:
                self.update_neighborhood(i - 1, closest_neuron_idx, current_data_point, learning_rate)
            if i < n - 1 and i % grid != 0:
                self.update_neighborhood(i + 1, closest_neuron_idx, current_data_point, learning_rate)
            if i >= grid:
                self.update_neighborhood(i - grid, closest_neuron_idx, current_data_point, learning_rate)
            if i < n - grid:
                self.update_neighborhood(i + grid, closest_neuron_idx, current_data_point, learning_rate)

    def train(self, step_by_step=False):
        if step_by_step:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))
        for epoch in range(self.epochs):
            np.random.shuffle(self.data)
            for x in self.data:
                closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
                self.weights[closest_neuron_idx] += self.learning_rate * (x - self.weights[closest_neuron_idx])
                self.find_neighborhood(closest_neuron_idx, x, self.learning_rate)
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

data, _ = make_blobs(n_samples=100, centers=5, n_features=5, random_state=42)
numnum = np.sqrt(data.shape[0])
num_neurons = int(max(numnum, 4))
epochs = 10
learning_rate = 0.3

som = SOM(data, num_neurons, epochs, learning_rate)
print(som.train(step_by_step=True))