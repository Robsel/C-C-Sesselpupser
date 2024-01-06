
import numpy as np
import matplotlib.pyplot as plt
import time

class SOM:
    def __init__(self, data, num_neurons, epochs, learning_rate):
        """
        Constructor for the Self-Organizing Map (SOM).
        
        Parameters:
        data (list of tuples): The input data, each tuple represents a data point
        num_neurons (int): Number of neurons in the SOM
        epochs (int): Number of epochs for training
        learning_rate (float): Initial learning rate
        """
        self.data = np.array(data)  # Convert data to NumPy array here
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.random.rand(num_neurons, self.data.shape[1])

    def train_and_visualize(self):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(8, 6))

        for epoch in range(self.epochs):
            np.random.shuffle(self.data)  # Shuffle data each epoch
            for x in self.data:
                closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
                self.weights[closest_neuron_idx] += self.learning_rate * (x - self.weights[closest_neuron_idx])

                # Visualization
                ax.clear()
                ax.scatter(self.data[:, 0], self.data[:, 1], label='Data Points')
                ax.scatter(self.weights[:, 0], self.weights[:, 1], marker='o', color='red', label='Neuron Positions')
                ax.set_title(f'SOM Training Epoch {epoch + 1}')
                ax.set_xlabel('Feature 1')
                ax.set_ylabel('Feature 2')
                ax.legend()
                plt.draw()
                plt.pause(0.1)  # Pause to update the plot

        plt.ioff()  # Turn off interactive mode

    def map_data(self):
        clusters = []
        for x in self.data:
            closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
            clusters.append(closest_neuron_idx)
        return clusters

def SOMofAwesom(data):
    num_neurons = max(len(data) // 2, 4)
    som = SOM(data, num_neurons, 10, 0.5)
    som.train_and_visualize()
    clusters = som.map_data()
    arrayplus1 = np.hstack((som.data, np.array(clusters).reshape(-1, 1)))
    return arrayplus1



data = ([0.87829205, 0.38761708, 0.15879426, 0.11401119, 0.22552955],
       [0.8090605 , 0.0731604 , 0.25829608, 0.12317188, 0.73187322],
       [0.00105176, 0.25615349, 0.63593292, 0.13331177, 0.68404951],
       [0.88622361, 0.99675417, 0.83636365, 0.89881699, 0.70979098],
       [0.02220191, 0.52333035, 0.55456961, 0.54287057, 0.10144522],
       [0.07797832, 0.90629405, 0.1625252 , 0.76118966, 0.89430632],
       [0.51497364, 0.84099041, 0.26260058, 0.44138421, 0.34699257],
       [0.56758333, 0.84500085, 0.74665355, 0.14449399, 0.63665848],
       [0.97658888, 0.73490274, 0.56283108, 0.94992819, 0.58821668],
       [0.20749093, 0.04312596, 0.06378419, 0.87609286, 0.48839679],
       [0.10855143, 0.19647104, 0.37930867, 0.01381685, 0.70843644],
       [0.36568419, 0.34122466, 0.59638539, 0.15849026, 0.35366594],
       [0.80203121, 0.91117115, 0.76662344, 0.79752782, 0.83639812],
       [0.52597151, 0.15189787, 0.15109721, 0.83766386, 0.9437445 ],
       [0.38651624, 0.65568798, 0.38983166, 0.81344063, 0.76742964],
       [0.84228686, 0.94687371, 0.01932561, 0.90344198, 0.16537699],
       [0.16018606, 0.23439486, 0.52358398, 0.9562685 , 0.20109191],
       [0.38251454, 0.10282365, 0.96335551, 0.2551877 , 0.42829581],
       [0.18899201, 0.74912394, 0.94916911, 0.75837967, 0.55855455],
       [0.04983727, 0.09939654, 0.30639166, 0.39036648, 0.7032137 ])
  
result = SOMofAwesom(data)
print(result)
# Example usage (assuming data is a 2D numpy array)
# data = np.array([...])
# result = SOMofAwesom(data)

    #def SOMofAwesom(array(n, d)):

        #SOMObj = SOM(array, int(max(array/2, 4)), 10, 0.5)
            #select random input
            #compute closest neuron
            #update neuron and neighbors
            #repeat for all input data
            #determine clusters based on nodes and their proximity to data points
            #save the clusters in a new arrayplus1(n, d+1), filled with array(n, d)

        #return arrayplus1