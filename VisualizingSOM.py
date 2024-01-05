import numpy as np
import matplotlib.pyplot as plt
from SOM import SOM  # Import the SOM class

class VisualizingSOM(SOM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot_som(self, data):
        fig, ax = plt.subplots()
        ax.set_xlim((0, self.som_dim[0]))
        ax.set_ylim((0, self.som_dim[1]))
        ax.set_title(f'SOM after {self.iteration} iterations')

        # Plot SOM nodes as small circles
        for x in range(self.som_dim[0]):
            for y in range(self.som_dim[1]):
                ax.plot(x, y, 'o', markersize=8, color='blue')

        # Draw lines to connect the SOM nodes
        for x in range(self.som_dim[0] - 1):
            for y in range(self.som_dim[1] - 1):
                ax.plot([x, x+1], [y, y], color='gray', linestyle='-', linewidth=1)
                ax.plot([x, x], [y, y+1], color='gray', linestyle='-', linewidth=1)

        # Plot input data points as dots
        for point in data:
            mapped = self._find_bmu(point)
            ax.plot(mapped[1], mapped[0], 'x', markersize=4, color='red')

        plt.show()

    def train_with_visualization(self, data, num_iterations, interval=10):
        for i in range(num_iterations):
            super().train(data, 1)  # Train for 1 iteration at a time
            if i % interval == 0:
                self.plot_som(data)

# Example Usage
input_data = np.random.rand(100, 5)  # 100 data points, each with 5 dimensions
visualizing_som = VisualizingSOM(input_dim=5, som_dim=(10, 10))
visualizing_som.train_with_visualization(input_data, num_iterations=100, interval=10)
