import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, data, grid_dimensions, epochs, learning_rate):
        self.data = data
        self.grid_dimensions = grid_dimensions
        self.epochs = epochs
        self.learning_rate = learning_rate

        # Determine the range for each dimension
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)

        # Create a structured grid for neuron initialization
        grid_x, grid_y = np.meshgrid(np.linspace(min_vals[0], max_vals[0], grid_dimensions[0]),
                                     np.linspace(min_vals[1], max_vals[1], grid_dimensions[1]))

        # Flatten the grid and replicate for all dimensions
        self.weights = np.zeros((np.prod(grid_dimensions), data.shape[1]))
        for i in range(data.shape[1] // 2):
            self.weights[:, i * 2] = grid_x.flatten()
            self.weights[:, i * 2 + 1] = grid_y.flatten()

        # Handle odd number of dimensions
        if data.shape[1] % 2 != 0:
            self.weights[:, -1] = np.mean([grid_x.flatten(), grid_y.flatten()], axis=0)

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
data = np.random.rand(100, 5)  # Example high-dimensional data
grid_dimensions = (10, 10)  # Grid size
som = SOM(data, grid_dimensions, epochs=100, learning_rate=0.1)
som.train(step_by_step=True)
