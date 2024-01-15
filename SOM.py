import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def initializegrid_onepoint(num_neurons, data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    #linspace_arrays = [np.linspace(min_vals[i], max_vals[i], num_neurons) for i in range(len(min_vals))]
    mid_points = (min_vals + max_vals) / 2
    return np.tile(mid_points, (num_neurons+1, 1))#np.column_stack(linspace_arrays)

def initializegrid_diagonal(num_neurons, data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    linspace_arrays = [np.linspace(min_vals[i], max_vals[i], num_neurons) for i in range(len(min_vals))]
    
    return np.column_stack(linspace_arrays)
    
class SOM:
    def __init__(self, data, num_neurons, epochs, learning_rate):
        self.data = np.array(data)
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights =  initializegrid_diagonal(num_neurons, self.data) #np.random.rand(num_neurons, self.data.shape[1])

    def update_neighborhood(self, neuron_idx, current_data_point, learning_rate):
        influence = 0.1
        self.weights[neuron_idx-1] += influence * learning_rate * (current_data_point - self.weights[neuron_idx-1])

    def find_neighborhood(self, closest_neuron_idx, current_data_point, learning_rate):
        n = num_neurons
        grid = max(int(np.sqrt(self.data.shape[0])), 4)
        
        if closest_neuron_idx > 0 and (closest_neuron_idx - 1) % grid != 0:
                self.update_neighborhood(closest_neuron_idx - 1, current_data_point, learning_rate)
        if closest_neuron_idx < n - 1 and closest_neuron_idx % grid != 0:
                self.update_neighborhood(closest_neuron_idx + 1, current_data_point, learning_rate)
        if closest_neuron_idx >= grid:
                self.update_neighborhood(closest_neuron_idx - grid, current_data_point, learning_rate)
        if closest_neuron_idx < n - grid:
                self.update_neighborhood(closest_neuron_idx + grid, current_data_point, learning_rate)

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

def analyze_array(arr):
    """
    Analyze a numpy array to determine:
    1. The length of the array (n)
    2. The number of unique elements in the array

    Parameters:
    arr (np.array): A numpy array of shape (n, 1)

    Returns:
    tuple: (length of the array, number of unique elements)
    """
    # Ensure the input is a numpy array
    if not isinstance(arr, np.ndarray):
        raise ValueError("Input must be a numpy array")

    # Check if the array shape is (n, 1)
    if arr.ndim != 2 or arr.shape[1] != 1:
        raise ValueError("Array must be of shape (n, 1)")

    # Calculate the length of the array
    length = arr.shape[0]

    # Calculate the number of unique elements
    unique_elements = np.unique(arr).size

    return length, unique_elements

data, _ = make_blobs(n_samples=120, centers=8, n_features=7, random_state=42)
numnum = np.sqrt(data.shape[0])
num_neurons = int(max(numnum, 4))
epochs = 10
learning_rate = 0.3

som = SOM(data, 8, epochs, learning_rate)
print(analyze_array(som.train(step_by_step=True)))