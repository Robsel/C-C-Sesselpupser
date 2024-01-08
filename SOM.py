import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


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

    def update_neighborhood(self, index, closest_neuron_idx, current_data_point, learning_rate):
        for i, weight in enumerate(self.weights):
            distance_to_best_neuron = np.linalg.norm(weight - self.weights[closest_neuron_idx])
            if distance_to_best_neuron <= self.radius:
                influence = np.exp(-distance_to_best_neuron**2 / (2 * (self.radius**2)))
                self.weights[i] += influence * learning_rate * (current_data_point - weight)

    def find_neighborhood(self, closest_neuron_idx, current_data_point, learning_rate):
        n, d = self.data.shape
        grid = int(np.sqrt(n))

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
            plt.ion()  # Enable interactive mode
            fig, ax = plt.subplots(figsize=(8, 6))

        for epoch in range(self.epochs):
            np.random.shuffle(self.data)
            for x in self.data:
                closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
                self.weights[closest_neuron_idx] += self.learning_rate * (x - self.weights[closest_neuron_idx])

                # Update neighboring neurons
                self.find_neighborhood(closest_neuron_idx, x, self.learning_rate)

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
                    plt.pause(0.05)  # Pause for a brief moment to update the plot

        if step_by_step:
            plt.ioff()  # Disable interactive mode
            print(self.map_data())
            

    def map_data(self):
        clusters = []
        for x in self.data:
            closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
            clusters.append(closest_neuron_idx)

        # Convert clusters to a numpy array and reshape it to be a column vector
        cluster_column = np.array(clusters).reshape(-1, 1)

        # Append the cluster column to the original data
        data_with_clusters = np.hstack((self.data, cluster_column))
        return data_with_clusters

# Usage
data, _ = make_blobs(n_samples=100, centers=4, n_features=5, random_state=42)

num_neurons = max(data.shape[0] // 2, 4)
epochs = 1
learning_rate = 0.1
radius = 0.1  # Radius of the neighborhood

som = SOM(data, num_neurons, epochs, learning_rate, radius)
som.train(step_by_step=True)  # Train with visualization but without the button
print



#give me code in python please, a function called find_neighborhood which gets a numpy array(n, d), checks what square is the next lower one to n and saves the root of that square in a variable called grid. then an itarator goes through a new numpy array(n, d), then chooses up to 4 other entries in the same numpy array based on the following criteria:
#is there an entry before this one? if yes->(check if this entry-1 equals 0 after modulo with grid. if yes, skip. if no, apply update_neighborhood to this entry-1) if no, skip.
#is there an entry after this one? if yes->(check if this entry equals 0 after modulo with grid. if yes, skip. if no, apply update_neighborhood to this entry+1) if no, skip.
#is there an entry at this-grid? if yes, apply update_neighborhood to this entry-grid. if no, skip.
#is there an entry at this+grid? if yes, apply update_neighborhood to this entry+grid. if no, skip.

