import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class SOM:
    def __init__(self, data, num_neurons, epochs, learning_rate, radius):
        self.data = np.array(data)
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.radius = radius
        self.weights = np.random.rand(num_neurons, self.data.shape[1])

    def update_neighborhood(self, best_neuron_idx, current_data_point, radius, learning_rate):
        for i, weight in enumerate(self.weights):
            distance_to_best_neuron = np.linalg.norm(weight - self.weights[best_neuron_idx])
            if distance_to_best_neuron <= radius:
                influence = np.exp(-distance_to_best_neuron**2 / (2 * (radius**2)))
                self.weights[i] += influence * learning_rate * (current_data_point - weight)

    def train(self, step_by_step=False):
        if step_by_step:
            plt.ion()  # Enable interactive mode
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.subplots_adjust(bottom=0.2)
            ax_button = plt.axes([0.7, 0.05, 0.1, 0.075])
            button = plt.Button(ax_button, 'Next Step')
            self.wait = True  # Control variable for button press

            def on_button_click(event):
                self.wait = False  # Resume loop on button click

            button.on_clicked(on_button_click)

        for epoch in range(self.epochs):
            np.random.shuffle(self.data)
            for x in self.data:
                closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
                self.weights[closest_neuron_idx] += self.learning_rate * (x - self.weights[closest_neuron_idx])

                # Update neighboring neurons
                self.update_neighborhood(closest_neuron_idx, x, self.radius, self.learning_rate)

                if step_by_step:
                    # Visualization at each step
                    ax.clear()
                    ax.scatter(self.data[:, 0], self.data[:, 1], label='Data Points')
                    ax.scatter(self.weights[:, 0], self.weights[:, 1], marker='o', color='red', label='Neuron Positions')
                    ax.set_title(f'SOM Training Epoch {epoch + 1}')
                    ax.set_xlabel('Feature 1')
                    ax.set_ylabel('Feature 2')
                    ax.legend()
                    plt.draw()

                    self.wait = True  # Wait for button press
                    while self.wait:
                        plt.pause(0.05)  # Pause for a brief moment to handle GUI events

        if step_by_step:
            plt.ioff()  # Disable interactive mode

    def map_data(self):
        clusters = []
        for x in self.data:
            closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
            clusters.append(closest_neuron_idx)
        return clusters

# Usage
data, _ = make_blobs(n_samples=100, centers=4, n_features=5, random_state=42)

num_neurons=max(data.shape[0]//2, 4)
epochs = 100
learning_rate = 0.5
radius = 0.1  # Radius of the neighborhood

som = SOM(data, num_neurons, epochs, learning_rate, radius)
som.train(step_by_step=True)
