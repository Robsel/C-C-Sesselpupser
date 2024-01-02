import numpy as np

class ElasticNet:
    def __init__(self, lr=0.01, n_iters=1000, l1_ratio=0.5, alpha=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            model = np.dot(X, self.weights) + self.bias
            dW = np.dot(X.T, (model - y)) / n_samples
            db = np.sum(model - y) / n_samples

            # L1 and L2 regularization terms
            l1_term = self.l1_ratio * self.alpha * np.sign(self.weights)
            l2_term = (1 - self.l1_ratio) * self.alpha * self.weights

            # Update weights and bias
            self.weights -= self.lr * (dW + l1_term + l2_term)
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Example usage:

# Generate some data
X, y = np.random.rand(100, 10), np.random.rand(100)

# Create ElasticNet model
model = ElasticNet(lr=0.01, n_iters=1000, l1_ratio=0.5, alpha=0.1)

# Fit model
model.fit(X, y)

# Predict
predictions = model.predict(X)
