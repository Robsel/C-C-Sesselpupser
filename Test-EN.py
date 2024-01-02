import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import ElasticNet

# Assuming the ElasticNet class is already defined as in the previous example

# Generate a synthetic dataset
X, y = np.random.rand(100, 10), np.random.rand(100)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the ElasticNet model
model = ElasticNet(lr=0.01, n_iters=1000, l1_ratio=0.5, alpha=0.1)
model.fit(X_train, y_train)

# Predict on the testing set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")
