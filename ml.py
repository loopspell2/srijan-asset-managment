import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

# Load the data
csv_path = 'predictive_maintenance_demo_data.csv'
df = pd.read_csv(csv_path)

# Preprocess the data
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
df['Last Maintenance Date'] = pd.to_datetime(df['Last Maintenance Date'])

# Feature engineering
df['Asset Age Days'] = (pd.to_datetime('today') - df['Purchase Date']).dt.days
df['Days Since Last Maintenance'] = (pd.to_datetime('today') - df['Last Maintenance Date']).dt.days

# Select features and target variable
features = ['Usage Hours', 'Maintenance Count', 'Environmental Temperature', 'Humidity', 'Asset Age Days', 'Days Since Last Maintenance']
target = 'Failure'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with a scaler and classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Example output for a single prediction
example_input = np.array([[5000, 5, 25, 50, 2000, 180]])  # Example feature values
example_pred = pipeline.predict(example_input)
example_pred_proba = pipeline.predict_proba(example_input)[:, 1]

print("Prediction (0=No Maintenance, 1=Maintenance Needed):", example_pred)
print("Prediction Probability:", example_pred_proba)


import joblib

# Save the model pipeline to a .pkl file
model_filename = 'predictive_maintenance_model.pkl'
joblib.dump(pipeline, model_filename)

print(f"Model saved to {model_filename}")
