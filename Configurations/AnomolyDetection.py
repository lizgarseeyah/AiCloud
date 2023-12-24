# Machine Learning (ML) for Anomaly Detection:

# Use Case: Inadequate logging and monitoring settings, lack of visibility.

# How AI Helps: ML algorithms can learn normal patterns of system behavior 
# and detect anomalies that may indicate a security incident. This can help 
# identify unusual API activities or unauthorized access.

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Generate some synthetic data for demonstration purposes
# In a real-world scenario, you would use actual data from your monitoring system
np.random.seed(42)
normal_data = np.random.normal(loc=0, scale=1, size=(1000, 10))  # Normal behavior
anomaly_data = np.random.uniform(low=-10, high=10, size=(50, 10))  # Anomalies

# Create the dataset by combining normal and anomaly data
X = np.vstack([normal_data, anomaly_data])
y = np.hstack([np.zeros(len(normal_data)), np.ones(len(anomaly_data))])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Isolation Forest model
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(X_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Convert predictions to binary (0: normal, 1: anomaly)
y_pred_binary = np.where(y_pred == -1, 1, 0)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("\nClassification Report:\n", classification_report(y_test, y_pred_binary))
