import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import label_binarize

# Load the data
X_test = pd.read_csv('X_test.csv')
Y_test = pd.read_csv('Y_test.csv').values.ravel()
x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(X_test)

# List of different n_estimators values to test
n_estimators_list = [50, 100, 150, 200, 250, 300]

# Initialize dictionary to store metrics for different n_estimators
metrics = {}

for n_estimators in n_estimators_list:
    # Initialize AdaBoost with the current number of estimators
    ada_boost = AdaBoostClassifier(n_estimators=n_estimators, random_state=42, algorithm='SAMME')
    
    # Fit AdaBoost model
    ada_boost.fit(X_train, y_train)  # Ensuring y_train is the correct shape

   # Predict the Cover Type
    y_pred = ada_boost.predict(X_test)
    y_pred_proba = ada_boost.predict_proba(X_test)

    # Calculate and store the accuracy, AUC, and F1 score
    accuracy = accuracy_score(Y_test, y_pred)
    auc = roc_auc_score(Y_test, y_pred_proba, multi_class='ovo', average='macro')  # Adjust for multiclass
    f1 = f1_score(Y_test, y_pred, average='macro')  # Adjust 'average' as per the task

    metrics[n_estimators] = {'Accuracy': accuracy, 'AUC': auc, 'F1 Score': f1}
    print(f'n_estimators={n_estimators}: Accuracy = {accuracy:.3f}, AUC = {auc:.3f}, F1 Score = {f1:.3f}')

# Optionally, you can analyze or plot the metrics to see the trend
import matplotlib.pyplot as plt

# Plotting the metrics
plt.figure(figsize=(10, 6))
estimators = list(metrics.keys())
accuracies = [metrics[n]['Accuracy'] for n in estimators]
aucs = [metrics[n]['AUC'] for n in estimators]
f1s = [metrics[n]['F1 Score'] for n in estimators]

plt.plot(estimators, accuracies, marker='o', label='Accuracy')
plt.plot(estimators, aucs, marker='o', label='AUC')
plt.plot(estimators, f1s, marker='o', label='F1 Score')
plt.xlabel('Number of Estimators')
plt.ylabel('Score')
plt.title('Performance Metrics vs. Number of Estimators in AdaBoost')
plt.legend()
plt.grid(True)
plt.show()
