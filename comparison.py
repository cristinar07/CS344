import argparse
import pandas as pd
from preprocess import standard_scale, minmax_range, add_irr_feature
from sklearn.neighbors import KNeighborsClassifier
import knn
from sklearn.naive_bayes import GaussianNB



def evaluate_knn(xTrain, yTrain, xTest, yTest):
    """
    Train a knn using xTrain and yTrain, and then predict
    the labels of the test data. This method will should
    return the classifier itself and the accuracy of the 
    resulting trained model on the test set.

    Parameters
    ----------
    xTrain : numpy nd-array with shape (n, d)
        Training data 
    yTrain : numpy 1d array with shape (n,)
        Array of labels associated with training data.
    xTest : numpy nd-array with shape (m, d)
        Test data 
    yTest : numpy 1d array with shape (m, )
        Array of labels associated with test data.

    Returns
    -------
    knn : an instance of the sklearn classifier associated with knn
        The knn model trained
    acc : float
        The accuracy of the trained model on the test data
    """
    xTrain, xTest = standard_scale(xTrain, xTest)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(xTrain, yTrain)
    yPred = knn.predict(xTest)
    correct_predictions = sum(yPred == yTest) 
    total_predictions = len(yTest) 
    acc = correct_predictions / total_predictions  
    
    return knn, acc



def evaluate_nb(xTrain, yTrain, xTest, yTest):
    """
    Train a knn using xTrain and yTrain, and then predict
    the labels of the test data. This method will should
    return the classifier itself and the accuracy of the 
    resulting trained model on the test set.

    Parameters
    ----------
    xTrain : numpy nd-array with shape (n, d)
        Training data 
    yTrain : numpy 1d array with shape (n,)
        Array of labels associated with training data.
    xTest : numpy nd-array with shape (m, d)
        Test data 
    yTest : numpy 1d array with shape (m, )
        Array of labels associated with test data.

    Returns
    -------
    knn : an instance of the sklearn classifier associated with knn
        The knn model trained
    acc : float
        The accuracy of the trained model on the test data
    """
    nb = GaussianNB()
    nb.fit(xTrain, yTrain)
    yPred = nb.predict(xTest)
    
    # Calculate the accuracy manually
    correct_predictions = sum(yPred == yTest) 
    total_predictions = len(yTest) 
    acc = correct_predictions / total_predictions  
    
    return nb, acc



def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="space_trainx.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="space_trainy.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="space_testx.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="space_testy.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    # flatten to compress to 1-d rather than (m, 1)
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()

    # add code here

    knn_model, knn_accuracy = evaluate_knn(xTrain, yTrain, xTest, yTest)
    
    xTrain_ss, xTest_ss = standard_scale(xTrain, xTest) #KNN Standard Scaling
    knn_model_ss, knn_accuracy_ss = evaluate_knn(xTrain_ss, yTrain, xTest_ss, yTest)
    
    xTrain_mm, xTest_mm = minmax_range(xTrain, xTest)# KNN Min-Max Scaling
    knn_model_mm, knn_accuracy_mm = evaluate_knn(xTrain_mm, yTrain, xTest_mm, yTest)

    xTrain_irr, xTest_irr = add_irr_feature(xTrain, xTest)#KNN with Irregular Features
    knn_model_irr, knn_accuracy_irr = evaluate_knn(xTrain_irr, yTrain, xTest_irr, yTest)

    

    nb_model, nb_accuracy = evaluate_nb(xTrain, yTrain, xTest, yTest)

    nb_model_ss, nb_accuracy_ss = evaluate_nb(xTrain_ss, yTrain, xTest_ss, yTest)# Naive Bayes Standard Scaling
    
    nb_model_mm, nb_accuracy_mm = evaluate_nb(xTrain_mm, yTrain, xTest_mm, yTest) #Naive Bayes Min-Max Scaling
  
    nb_model_irr, nb_accuracy_irr = evaluate_nb(xTrain_irr, yTrain, xTest_irr, yTest)   #Naive Bayes Irregular Features

    # Print the results
    print("Preprocessing Technique   | K-NN Accuracy   | Naive Bayes Accuracy")
    print(f"No Preprocessing          | {knn_accuracy:.6f}        | {nb_accuracy:.6f}")
    print(f"Irregular Features        | {knn_accuracy_irr:.6f}        | {nb_accuracy_irr:.6f}")
    print(f"Min-Max Scaling           | {knn_accuracy_mm:.6f}        | {nb_accuracy_mm:.6f}")
    print(f"Standard Scaling          | {knn_accuracy_ss:.6f}        | {nb_accuracy_ss:.6f}")
    

if __name__ == "__main__":
    main()
