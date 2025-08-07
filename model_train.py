import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Sample user data (you can replace this with real data later)
data = pd.DataFrame({
    'age': [22, 35, 26, 29, 45, 50, 31, 40, 24, 38],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male'],
    'job': ['Engineer', 'Doctor', 'Student', 'Lawyer', 'Teacher', 'Engineer', 'Lawyer', 'Doctor', 'Student', 'Engineer'],
    'income': [30000, 60000, 15000, 50000, 40000, 70000, 55000, 62000, 12000, 68000],
    'marital_status': ['Single', 'Married', 'Single', 'Single', 'Married', 'Married', 'Single', 'Married', 'Single', 'Married'],
    'product_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
})

# Encoding categorical variables
data_encoded = pd.get_dummies(data.drop('product_id', axis=1))
labels = data['product_id']

X_train, X_test, y_train, y_test = train_test_split(data_encoded, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('job_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the column names for encoding during prediction
with open('product_mapping.pkl', 'wb') as f:
    pickle.dump(data_encoded.columns.tolist(), f)

print("✅ Model training complete. Files saved.")
