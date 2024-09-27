import pandas as pd
import joblib

# Load the model
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

test_data = pd.read_csv('test.csv')
input_data = test_data.drop(columns='status')
actuals = test_data['status']

# Scale the input data
input_scaled = scaler.transform(input_data)
# Make predictions
predictions = model.predict(input_scaled)
probabilities = model.predict_proba(input_scaled)

# Output results
for i in range(len(predictions)):
    predicted_status = predictions[i]
    accuracy = probabilities[i].max()
    print(f"Sample {i+1}: Predicted: {predicted_status}, Accuracy: {accuracy:.2f}, Actual: {actuals[i]}")