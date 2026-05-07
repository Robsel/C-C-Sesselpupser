import numpy as np
import sys
from enum import Enum
from collections import Counter
import time

class Init_Mode(Enum):
    diagonal = 'diagonal'
    onepoint = 'onepoint'
    random = 'random'

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

class SOM:
    """
    Implements a Self Organizing Map
        
        Attributes
        ----------
        data : arraylike
            your data points
        num_neurons : int
            The number of neurons (best is the number of expected clusters)
        epochs : int
            The number of epochs this algorithm will run
        learning_rate : float
            Describes how much new data points will change the current model
        influence : float
            Describes how much a neuron will update its neighbors when its changed
        update_neighbors_epoch : int
            After how many epochs should a neuron also update its neighbors
        calculate_k_epoch : int
            After how many epochs should each neuron calculate a new neighborhood
        k_neighbors : int
            How many neighbors can a neuron have
        init_mode : Init_Mode
            How should the neurons be initialized?
            Possible values are diagonal, onepoint and random
    """

    def __init__(self, data, num_neurons: int , epochs: int = 100, learning_rate: float = 0.3, influence: float = 0.1, update_neighbors_epoch: int = 4, calculate_k_epoch: int = 6, first_k_calc = True, k_neighbors: int = 4, init_mode: Init_Mode='diagonal'):
        """
        Initialize a new Self Organizing Map
        
        Parameters
        ----------
            data
                your data points
            num_neurons : int
                The number of neurons (best is the number of expected clusters)
            epochs : int
                The number of epochs this algorithm will run (default is 100)
            learning_rate : float
                Describes how much new data points will change the current model (default is 0.3)
            influence : float
                Describes how much a neuron will update its neighbors when its changed (default is 0.1)
            update_neighbors_epoch : int
                After how many epochs should a neuron also update its neighbors (default is 4)
            calculate_k_epoch : int
                After how many epochs should each neuron calculate a new neighborhood (default is 6)
            first_k_calc: boolean
                If True, calculates nearest neighbors on first call of train (default is True)
            k_neighbors : int
                How many neighbors can a neuron have (default is 4)
            init_mode : Init_Mode
                How should the neurons be initialized?
                Possible values are diagonal, onepoint and random (default ist diagonal)
        """
        self.data = np.array(data)
        self.num_neurons = num_neurons
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.influence = influence
        self.update_neighbors_epoch = update_neighbors_epoch
        self.calculate_k_epoch = calculate_k_epoch
        self.first_k_calc = first_k_calc
        self.k_neighbors = k_neighbors
        self.init_mode = init_mode

        if init_mode == 'diagonal':
            self.weights = initializegrid_diagonal(num_neurons, self.data)
        elif init_mode == 'onepoint':
            self.weights = initializegrid_onepoint(num_neurons, self.data)
        elif init_mode == 'random':
            self.weights = initialize_random(num_neurons, self.data)
        else:
            raise ValueError("Invalid init_mode. Choose 'diagonal', 'onepoint', or 'random'")
        self.neighbors = {i: [] for i in range(num_neurons)}

    def calculate_k_closest_neighbors(self):
        distances = np.linalg.norm(self.weights[:, None] - self.weights, axis=2)
        for i in range(self.num_neurons):
            self.neighbors[i] = np.argsort(distances[i])[:self.k_neighbors]

    def update_neighborhood(self, neuron_idx, current_data_point, learning_rate, influence):
        neighbor_indices = self.neighbors[neuron_idx]
        self.weights[neighbor_indices] += influence * learning_rate * (current_data_point - self.weights[neighbor_indices])

    def train(self):
        if self.first_k_calc:
            self.calculate_k_closest_neighbors()
            self.first_k_calc = False

        for epoch in range(self.epochs):
            # Shuffle data each epoch
            np.random.shuffle(self.data)
            
            for x in self.data:
                closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
                self.weights[closest_neuron_idx] += self.learning_rate * (x - self.weights[closest_neuron_idx])

                if epoch >= self.update_neighbors_epoch and epoch % self.update_neighbors_epoch == 0:
                    self.update_neighborhood(closest_neuron_idx, x, self.learning_rate, self.influence)

            if epoch >= self.calculate_k_epoch and epoch % self.calculate_k_epoch == 0:
                self.calculate_k_closest_neighbors()

        return self.map_data()

    def map_data(self):
        clusters = []
        for x in self.data:
            closest_neuron_idx = np.argmin(np.linalg.norm(self.weights - x, axis=1))
            clusters.append(closest_neuron_idx)
        return clusters

# --- Helper: load data from file ---
def load_data(filename):
    data = []
    try:
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    row = list(map(float, line.split()))
                    if row:
                        data.append(row)
    except FileNotFoundError:
        print(f"Error: Could not open file {filename}")
        sys.exit(1)
    
    if not data:
        print("Error: No valid data found in file.")
        sys.exit(1)
    
    return np.array(data)

# --- Helper: read input with defaults ---
def read_int(prompt, default_val):
    try:
        user_input = input(f"{prompt} [{default_val}]: ").strip()
        if user_input == "":
            return default_val
        return int(user_input)
    except ValueError:
        return default_val

def read_float(prompt, default_val):
    try:
        user_input = input(f"{prompt} [{default_val}]: ").strip()
        if user_input == "":
            return default_val
        return float(user_input)
    except ValueError:
        return default_val

# --- MAIN ---
if __name__ == "__main__":
    # Check if data file was provided
    if len(sys.argv) < 2:
        print("Error: No data file provided.")
        print("Usage: python newSOM.py <datafile>")
        sys.exit(1)
    
    filename = sys.argv[1]
    data = load_data(filename)
    
    # Default parameters
    default_num_neurons = 10
    default_epochs = 100
    default_learning_rate = 0.3
    default_influence = 0.1
    default_update_neighbors_epoch = 4
    default_calculate_k_epoch = 6
    default_k_neighbors = 4
    default_init_mode = 0
    
    # Read parameters from user
    num_neurons = read_int("Enter number of neurons", default_num_neurons)
    epochs = read_int("Enter number of epochs", default_epochs)
    learning_rate = read_float("Enter learning rate", default_learning_rate)
    influence = read_float("Enter influence", default_influence)
    update_neighbors_epoch = read_int("Enter update_neighbors_epoch", default_update_neighbors_epoch)
    calculate_k_epoch = read_int("Enter calculate_k_epoch", default_calculate_k_epoch)
    k_neighbors = read_int("Enter k_neighbors", default_k_neighbors)
    init_mode_input = read_int("Init mode (0 = diagonal, 1 = onepoint, 2 = random)", default_init_mode)
    
    # Convert init mode
    init_mode_map = {0: 'diagonal', 1: 'onepoint', 2: 'random'}
    init_mode = init_mode_map.get(init_mode_input, 'diagonal')
    
    # Create and train SOM
    som = SOM(data,
              num_neurons,
              epochs,
              learning_rate,
              influence,
              update_neighbors_epoch,
              calculate_k_epoch,
              k_neighbors,
              init_mode=init_mode)
    
    start = time.perf_counter()
    clusters = som.train()
    end = time.perf_counter()
    print(f"\nTraining completed in {end - start} seconds.")
    # Output results
    print("\nCluster assignments:")
    cluster_counts = Counter(clusters)
    
    for cluster_id in sorted(cluster_counts.keys()):
        print(f"{cluster_id}: {cluster_counts[cluster_id]}")
    
    