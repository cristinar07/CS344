import argparse
import numpy as np
import pandas as pd


class Knn(object):
    k = 0              # number of neighbors to use
    nFeatures = 0      # number of features seen in training
    nSamples = 0       # number of samples seen in training
    isFitted = False  # has train been called on a dataset?


    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k

    def train(self, xFeat, y): #xFeat, diffrent feature, could more than two features, y represent the lables. 
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (n, d)
            Training data 
        y : numpy 1d array with shape (n, )
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
       
        if xFeat.shape[0] != y.shape[0]:
            raise ValueError("Shape of the data does not match")

        self.xTrain = xFeat
        self.yTrain = y
        return self
        


    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : numpy nd-array with shape (m, d)
            The data to predict.  

        Returns
        -------
        yHat : numpy.1d array with shape (m, )
            Predicted class label per sample
        """
        yHat = []
        for x in xFeat:
            distances = np.sqrt(np.sum((self.xTrain - x) ** 2, axis=1))
            nearest_neighbors_indices = np.argsort(distances)[:self.k]
            nearest_neighbors_labels = self.yTrain[nearest_neighbors_indices]
            most_common_label = np.argmax(np.bincount(nearest_neighbors_labels.astype(int)))

            yHat.append(most_common_label)

        return np.array(yHat) 




def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape (n,)
        Predicted class label for n samples
    yTrue : 1d-array with shape (n, )
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    yHat = np.array(yHat)
    yTrue = np.array(yTrue)
    rightpredictions = np.sum(yHat == yTrue)

    acc = rightpredictions / float(len(yTrue))

    return acc

def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="simxTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="simyTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="simxTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="simyTest.csv",
                        help="filename for labels associated with the test data")
    

    args = parser.parse_args()
    # load the train and test data
    # assume the data is all numerical and 
    # no additional pre-processing is necessary
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain)
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest)
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

if __name__ == "__main__":
    main()
