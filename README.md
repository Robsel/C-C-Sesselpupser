Self-Organizing Map (SOM) Implementation in Python

This repository contains an implementation of a Self-Organizing Map (SOM), a type of artificial neural network that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional) representation of the input space. This implementation is particularly useful for visualizing high-dimensional data in a lower-dimensional space.
Features

    Initialization Modes: Diagonal, One Point, Random.
    Adjustable Parameters: Learning rate, neighborhood influence, and number of neurons.
    Dynamic Neighborhood Updating: Automatic updating during training.
    Visualization Support: Visualize the training process in real-time.
    Data Mapping: Mapping of data points to neurons for cluster identification.

Dependencies

    numpy
    matplotlib
    sklearn (for example data generation)

Installation

No specific installation steps are required apart from ensuring the required dependencies are installed. You can install the dependencies using pip:

bash

pip install numpy matplotlib scikit-learn

Usage
Step 1: Import the SOM class and Auxiliary Functions

python

from som_module import SOM, analyze_array

Step 2: Prepare Your Data

python

from sklearn.datasets import make_blobs
data, _ = make_blobs(n_samples=1000, centers=50, n_features=7, random_state=42)

Step 3: Initialize the SOM

python

num_neurons = int(np.ceil(np.sqrt(data.shape[0])))
epochs = 100
learning_rate = 0.3
update_neighbors_epoch = 5
influence = 0.1
calculate_k_epoch = 2
k_neighbors = 5

som = SOM(data, num_neurons, epochs, learning_rate, influence, update_neighbors_epoch, calculate_k_epoch, k_neighbors, randomize_data=False, init_mode='diagonal')

Step 4: Train the SOM

python

mapped_data = som.train(step_by_step=True)

Step 5: Analyze the Results

python

print(analyze_array(mapped_data))

Step 6: Visualizing the SOM (Optional)

Set step_by_step to True to enable real-time training visualization.
