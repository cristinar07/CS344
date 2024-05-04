import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load the data
X_test = pd.read_csv('X_test.csv')
Y_test = pd.read_csv('y_test.csv').values.ravel()
x_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(X_test)

# List of different n_estimators values to test
n_estimators_list = [50, 100, 150, 200, 250, 300]

# Initialize dictionary to store metrics for different n_estimators
metrics = {}

# ada_boost = AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME')
# ada_boost.fit(X_train, y_train)
# y_pred = ada_boost.predict(X_test)
# y_prey_pred_proba = ada_boost.predict_proba(X_test)[:, 1]
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
    print(f'n_estimators={n_estimators}: Accuracy = {accuracy:.4f}, AUC = {auc:.4f}, F1 Score = {f1:.4f}')

# Optionally, you can analyze or plot the metrics to see the trend
# import matplotlib.pyplot as plt

# # Binarize the output classes for the multiclass case
# y_test_binarized = label_binarize(Y_test, classes=np.unique(Y_test))
# n_classes = y_test_binarized.shape[1]

# # Calculate probabilities for each class
# y_pred_proba = ada_boost.predict_proba(X_test)

# # Compute ROC curve and ROC area for each class
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
#     roc_auc[i] = auc(fpr[i], tpr[i])

# # Compute micro-average ROC curve and ROC area
# fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred_proba.ravel())
# roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # Plot all ROC curves
# plt.figure(figsize=(8, 6))

# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

# plt.plot(fpr["micro"], tpr["micro"],
#          label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
#          color='deeppink', linestyle=':', linewidth=4)

# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC of AdaBoost')
# plt.legend(loc="lower right")
# plt.show()