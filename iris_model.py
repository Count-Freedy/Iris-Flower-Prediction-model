import pandas as pd

# Load dataset
df = pd.read_csv("IRIS.csv")  # Ensure the dataset is in the same folder as this script

# Display first 5 rows
print(df.head())

# Check for missing values
print("\nMissing Values:\n", df.isnull().sum())

# Display dataset statistics
print("\nDataset Statistics:\n", df.describe())

# Display unique species
print("\nUnique Species:\n", df['species'].value_counts())

from sklearn.preprocessing import LabelEncoder

# Encode species column
encoder = LabelEncoder()
df['species'] = encoder.fit_transform(df['species'])

# Check encoding
print("\nDataset after encoding:\n", df.head())

from sklearn.model_selection import train_test_split

# Define Features (X) and Target (y)
X = df.drop(columns=['species'])  # Features (sepal & petal sizes)
y = df['species']  # Target (species)

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print split sizes
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

import joblib

# Save the trained model
joblib.dump(model, "iris_model.pkl")

print("\nModel saved successfully as 'iris_model.pkl'!")
