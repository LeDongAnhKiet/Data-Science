import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve

# Load data
creditcard_data = pd.read_csv('creditcard.csv')

# Display data info
print(creditcard_data.shape)
print(creditcard_data.head())

# Class distribution
print("Class distribution: \n", creditcard_data['Class'].value_counts())

# Summary statistics of 'Amount'
print("Summary statistics of 'Amount':\n", creditcard_data['Amount'].describe())

# Scaling 'Amount' and removing 'Time'
scaler = StandardScaler()
creditcard_data['Amount'] = scaler.fit_transform(creditcard_data[['Amount']])
NewData = creditcard_data.drop(columns=['Time'])

# Splitting dataset into training and testing sets
X = NewData.drop('Class', axis=1)
y = NewData['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Logistic Regression Model
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Model evaluation
logistic_predictions = logistic_model.predict_proba(X_test)[:, 1]
logistic_auc = roc_auc_score(y_test, logistic_predictions)
print(f"Logistic Regression AUC: {logistic_auc}")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, logistic_predictions)
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {logistic_auc:.2f})', color='blue')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Decision Tree Model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

# Decision Tree predictions and plot
tree_predictions = decision_tree.predict(X_test)
tree_auc = roc_auc_score(y_test, decision_tree.predict_proba(X_test)[:, 1])

print(f"Decision Tree AUC: {tree_auc}")
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, filled=True, feature_names=X.columns, class_names=['0', '1'], proportion=True)
plt.show()

# Neural Network Model
nn_model = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=123)
nn_model.fit(X_train, y_train)

# Neural network predictions
nn_predictions = nn_model.predict(X_test)
nn_auc = roc_auc_score(y_test, nn_model.predict_proba(X_test)[:, 1])

print(f"Neural Network AUC: {nn_auc}")

# Gradient Boosting Model (using XGBoost)
xgb_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.01, max_depth=3, subsample=0.5, colsample_bytree=0.8, random_state=123)
xgb_model.fit(X_train, y_train)

# XGBoost predictions and AUC calculation
xgb_predictions = xgb_model.predict_proba(X_test)[:, 1]
xgb_auc = roc_auc_score(y_test, xgb_predictions)

print(f"XGBoost AUC: {xgb_auc}")

# Plot ROC for XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_predictions)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_auc:.2f})', color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Model performance comparison (AUCs)
print(f"Logistic Regression AUC: {logistic_auc}")
print(f"Decision Tree AUC: {tree_auc}")
print(f"Neural Network AUC: {nn_auc}")
print(f"XGBoost AUC: {xgb_auc}")
