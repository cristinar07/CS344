import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Load the data
X_test = pd.read_csv('X_test.csv')
Y_test = pd.read_csv('Y_test.csv').values.ravel()  # Ensuring Y_test is a 1D array
x_train = pd.read_csv('x_train.csv')
y_train = pd.read_csv('y_train.csv').values.ravel()  # Ensuring y_train is a 1D array

# Standardize the numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_test = scaler.transform(X_test)

# Define different configurations for hidden layer sizes
hidden_layer_sizes = [
    (50,),         # One hidden layer with 50 neurons
    (100,),        # One hidden layer with 100 neurons
    (50, 30),      # Two hidden layers with 50 and 30 neurons 
    (100, 50),     # Two hidden layers with 100 and 50 neurons
    (100, 50, 25)  # Three hidden layers with 100, 50, and 25 neurons
]

# Dictionary to store results
results = {}

for size in hidden_layer_sizes:
    mlp = MLPClassifier(hidden_layer_sizes=size, max_iter=300, activation='relu', solver='adam', random_state=42)
    
    # Train the model
    mlp.fit(X_train, y_train)

    # Make predictions
    y_pred = mlp.predict(X_test)
    y_pred_proba = mlp.predict_proba(X_test)

    # Calculate the performance metrics
    accuracy = accuracy_score(Y_test, y_pred)
    auc = roc_auc_score(Y_test, y_pred_proba, multi_class='ovo', average='macro')
    f1 = f1_score(Y_test, y_pred, average='macro')

    # Store results
    results[size] = {'Accuracy': accuracy, 'AUC': auc, 'F1 Score': f1}

# Output the results
for size, metrics in results.items():
    print(f"Hidden Layers: {size} -> Accuracy: {metrics['Accuracy']:.3f}, AUC: {metrics['AUC']:.3f}, F1 Score: {metrics['F1 Score']:.3f}")
