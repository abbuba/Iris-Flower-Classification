# 1. Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- Step 1: Load the Dataset ---
print("Step 1: Loading the Iris dataset...")
iris = load_iris()

# Create a pandas DataFrame for easier manipulation
# iris.data contains the measurements, iris.feature_names contains the column names
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
# Add the species (target) to the DataFrame
# iris.target contains the numeric labels (0, 1, 2)
df['species'] = iris.target
print("Dataset loaded successfully.\n")


# --- Step 2: Explore the Data Visually ---
print("Step 2: Exploring the data...")
# Use seaborn's pairplot to visualize the relationships between features
# The 'hue' parameter colors the points by species
print("Displaying pairplot... Close the plot window to continue.")
sns.pairplot(df, hue='species', palette='viridis')
plt.show() # This will pause the script until you close the plot window
print("Exploration complete.\n")


# --- Step 3: Split the Data ---
print("Step 3: Splitting data into training and testing sets...")
# X contains the features (the four measurements)
X = iris.data
# y contains the labels (the species)
y = iris.target

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples\n")


# --- Step 4: Train the K-Nearest Neighbors (KNN) Model ---
print("Step 4: Training the K-Nearest Neighbors model...")
# Initialize the classifier with k=3 (a common choice for k)
knn = KNeighborsClassifier(n_neighbors=3)

# Train the model using the training data
knn.fit(X_train, y_train)
print("Model training complete.\n")


# --- Step 5: Evaluate the Model ---
print("Step 5: Evaluating the model...")
# Make predictions on the unseen test data
y_pred = knn.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"--- Accuracy: {accuracy:.4f} ---")

# Display the confusion matrix
print("\n--- Confusion Matrix ---")
# A confusion matrix shows the number of correct and incorrect predictions per class
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Display the classification report
print("\n--- Classification Report ---")
# This report shows the main classification metrics: precision, recall, and f1-score
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)
