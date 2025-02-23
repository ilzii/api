import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate synthetic data
data = {
    'Mileage': np.random.randint(1000, 100000, n_samples),
    'Car_Age': np.random.randint(1, 10, n_samples),
    'Car_Model': np.random.choice(['Toyota Corolla', 'Honda Civic', 'Ford Focus', 'Hyundai Elantra', 'Chevrolet Malibu', 'Nissan Altima', 'Kia Optima', 'Volkswagen Jetta', 'Subaru Outback', 'Mazda CX-5'], n_samples),
    'Last_Service_Date': [datetime.today() - timedelta(days=np.random.randint(30, 365*3)) for _ in range(n_samples)],
    'Driving_Conditions': np.random.choice(['Normal', 'Rainy', 'Hot', 'Cold'], n_samples),
    'Recent_Issues': np.random.choice(['None', 'Brake Issue', 'Engine Noise', 'Battery Issue', 'Transmission', 'Tire Issue'], n_samples),
}

# Create a DataFrame
df = pd.DataFrame(data)

# Define a function to determine maintenance need
def maintenance_need(row):
    if (row['Mileage'] > 50000 or
        row['Car_Age'] > 5 or
        row['Recent_Issues'] != 'None' or
        (datetime.today() - row['Last_Service_Date']).days > 365):
        return 1
    return 0

# Apply the function to create the target column
df['Maintenance_Needed'] = df.apply(maintenance_need, axis=1)

# Save the dataset to a CSV file
df.to_csv('predictive_maintenance_dataset.csv', index=False)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('predictive_maintenance_dataset.csv')

# Encode categorical features
df = pd.get_dummies(df, columns=['Car_Model', 'Driving_Conditions', 'Recent_Issues'], drop_first=True)

# Define features (X) and target (y)
X = df.drop('Maintenance_Needed', axis=1)
y = df['Maintenance_Needed']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))