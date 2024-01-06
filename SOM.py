import numpy as np

class SOM:
    def __init__(self, data, num_neurons, epochs, learning_rate):
        """
        Constructor for the Self-Organizing Map (SOM).
        
        Parameters:
        data (numpy.ndarray): The input data, shape (n, d)
        num_neurons (int): Number of neurons in the SOM
        max_epochs (int): Maximum number of epochs for training
        learning_rate (float): Initial learning rate
        """
        self.data = data
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate

    def train(self):
        for counter in range(self.epochs):
            for x in self.data:
                # Find the neuron that is closest to the input data point
                closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
                # Update the weights of the neuron to be closer to the input data point
                self.weights[closest_neuron_idx] += self.learning_rate * (x - self.weights[closest_neuron_idx])

    def map_data(self):
        clusters = []
        for x in self.data:
            # Find the neuron that is closest to the input data point
            closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
            clusters.append(closest_neuron_idx)
        return clusters

    def SOMofAwesom(data):
        num_neurons = max(data.shape[0] // 2, 4)
        som = SOM(data, num_neurons, 10, 0.5)
        som.train()
        clusters = som.map_data()
        arrayplus1 = np.hstack((data, np.array(clusters).reshape(-1, 1)))

        return arrayplus1

    #def SOMofAwesom(array(n, d)):

        #SOMObj = SOM(array, int(max(array/2, 4)), 10, 0.5)
            #select random input
            #compute closest neuron
            #update neuron and neighbors
            #repeat for all input data
            #determine clusters based on nodes and their proximity to data points
            #save the clusters in a new arrayplus1(n, d+1), filled with array(n, d)

        #return arrayplus1