import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

# Load the dataset
customer_data = pd.read_csv('Mall_Customers.csv')

# View the structure of the dataset
print(customer_data.info())

# Display column names
print(customer_data.columns)

# Display first few records
print(customer_data.head())

# Descriptive statistics for 'Age'
print(customer_data['Age'].describe())
print('Standard Deviation of Age:', np.std(customer_data['Age']))

# Descriptive statistics for 'Annual Income (k$)'
print(customer_data['Annual Income (k$)'].describe())
print('Standard Deviation of Annual Income:', np.std(customer_data['Annual Income (k$)']))

# Descriptive statistics for 'Spending Score (1-100)'
print(customer_data['Spending Score (1-100)'].describe())
print('Standard Deviation of Spending Score:', np.std(customer_data['Spending Score (1-100)']))

# Bar plot for gender comparison
sns.countplot(x='Gender', data=customer_data, hue='Gender', palette='rainbow', dodge=False, legend=False)
plt.title('Gender Comparison')
plt.ylabel('Count')
plt.xlabel('Gender')
plt.show()

# Pie chart for gender comparison
gender_counts = customer_data['Gender'].value_counts()
plt.pie(gender_counts, labels=['Female', 'Male'], autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Pie Chart of Gender Distribution')
plt.show()

# Histogram for Age
customer_data['Age'].plot(kind='hist', bins=20, color='blue', edgecolor='black')
plt.title('Histogram for Age')
plt.xlabel('Age Class')
plt.ylabel('Frequency')
plt.show()

# Boxplot for Age
sns.boxplot(customer_data['Age'], color='pink')
plt.title('Boxplot for Age')
plt.show()

# Histogram for Annual Income
customer_data['Annual Income (k$)'].plot(kind='hist', bins=20, color='#660033', edgecolor='black')
plt.title('Histogram for Annual Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Frequency')
plt.show()

# Density plot for Annual Income
sns.kdeplot(customer_data['Annual Income (k$)'], fill=True, color='yellow')
plt.title('Density Plot for Annual Income')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Density')
plt.show()

# Boxplot for Spending Score
sns.boxplot(customer_data['Spending Score (1-100)'], color='#990000')
plt.title('Boxplot for Spending Score')
plt.show()

# Histogram for Spending Score
customer_data['Spending Score (1-100)'].plot(kind='hist', bins=20, color='#6600cc', edgecolor='black')
plt.title('Histogram for Spending Score')
plt.xlabel('Spending Score (1-100)')
plt.ylabel('Frequency')
plt.show()

# K-means clustering and Elbow method
X = customer_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method to determine the optimal number of clusters
kmeans = KMeans()
visualizer = KElbowVisualizer(kmeans, k=(1,11))
visualizer.fit(X_scaled)
visualizer.show()

# Silhouette analysis for different values of K
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(X_scaled)
    visualizer.show()

# Optimal clustering based on Elbow or Silhouette method
optimal_k = 6  # Example optimal number from analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA for dimensionality reduction (2 components for visualization)
from sklearn.decomposition import PCA
pca = PCA(2)
X_pca = pca.fit_transform(X_scaled)

# Plotting clusters with PCA-reduced components
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=customer_data['Cluster'], palette='Set2', s=100, alpha=0.7)
plt.title('K-means Clustering with PCA')
plt.show()

# Scatter plot of clusters based on Annual Income and Spending Score
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', data=customer_data, palette='Set1', s=100, alpha=0.7)
plt.title('K-means Clustering based on Annual Income and Spending Score')
plt.show()

# Scatter plot of clusters based on Spending Score and Age
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Spending Score (1-100)', y='Age', hue='Cluster', data=customer_data, palette='Set1', s=100, alpha=0.7)
plt.title('K-means Clustering based on Spending Score and Age')
plt.show()
