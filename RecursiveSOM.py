import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from SOM import SOM  # Ensure this is correctly imported

class RecursiveSOM(SOM):
    
    def __init__(self, data, num_neurons, epochs, learning_rate, a, b):
        super().__init__(data, num_neurons, epochs, learning_rate)
        self.a = a
        self.b = b
        self.wy = np.random.rand(num_neurons, data.shape[1])
        self.y = np.zeros((num_neurons, data.shape[1]))

    def calculate_error(self, x, y_prev):
        e_x = np.linalg.norm(x - self.weights, axis=1)**2
        e_y = np.linalg.norm(y_prev - self.wy, axis=1)**2
        E = self.a * e_x + self.b * e_y
        return E

    def transfer_function(self, E):
        return np.exp(-E)

    def update_weights(self, x, y_prev, closest_neuron_idx):
        # Update feed-forward weights
        super().update_neighborhood(closest_neuron_idx, closest_neuron_idx, x, self.learning_rate)

        # Update recurrent weights
        y_prev_neuron = y_prev[closest_neuron_idx]
        self.wy[closest_neuron_idx] += self.learning_rate * (y_prev_neuron - self.wy[closest_neuron_idx])

    def train(self, step_by_step=False):
        if step_by_step:
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 6))

        for epoch in range(self.epochs):
            y_prev = np.zeros((self.num_neurons, self.data.shape[1]))
                              
            np.random.shuffle(self.data)
            for x in self.data:
                E = self.calculate_error(x, y_prev)
                closest_neuron_idx = np.argmin(E)
                self.update_weights(x, y_prev, closest_neuron_idx)
                self.y[closest_neuron_idx] = self.transfer_function(E[closest_neuron_idx])

            if step_by_step:
                    ax.clear()
                    ax.scatter(self.data[:, 0], self.data[:, 1], label='Data Points')
                    ax.scatter(self.weights[:, 0], self.weights[:, 1], marker='o', color='red', label='Neuron Positions')
                    ax.set_title(f'Recursive SOM Training Epoch {epoch + 1}')
                    ax.set_xlabel('Feature 1')
                    ax.set_ylabel('Feature 2')
                    ax.legend()
                    plt.draw()
                    plt.pause(0.05)

        if step_by_step:
            plt.ioff()

# Sample usage
data, _ = make_blobs(n_samples=100, centers=4, n_features=5, random_state=42)
epochs = 1
learning_rate = 0.3
a, b = 1.0, 1.0
numnum = np.sqrt(data.shape[0])
numnum *= numnum
num_neurons = int(max(numnum , 4))

recursive_som = RecursiveSOM(data, num_neurons, epochs, learning_rate, a, b)
recursive_som.train(step_by_step=True)
