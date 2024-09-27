import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Read the data
df1 = pd.read_csv('dataset\parkinsons.data')
df2 = pd.read_csv('dataset\parkinsons_updrs.data')
common = df1.columns.intersection(df2.columns)
df = pd.concat([df1[common], df2[common]], ignore_index=True)

# Tạo DataFrame ngẫu nhiên
pdf = pd.DataFrame(df.describe())
def generate_random_data(attr_stats, n_samples=10):
    return np.random.normal(loc=attr_stats['mean'], scale=attr_stats['std'], size=n_samples).clip(
        attr_stats['min'], attr_stats['max'])
generated_df = pd.DataFrame({col: generate_random_data(attr_stats) for col, attr_stats in pdf.items()})
generated_df['status'] = generated_df['status'].round().astype(int)
generated_df.to_csv('test.csv', index=False)

# Get the features and labels
features = df.drop(columns=['status', 'name']).values
labels = df['status'].values

# Scale the features to between -1 and 1
scaler = MinMaxScaler((-1, 1))
x = scaler.fit_transform(features)
y = labels

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Train the model
model = XGBClassifier(eta=0.1, objective='binary:logistic')
model.fit(x_train, y_train)

# Calculate the accuracy
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Save model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
