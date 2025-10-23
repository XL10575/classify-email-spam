# -*- coding: utf-8 -*-
"""
Demo of 10-fold cross-validation using Gaussian naive Bayes on spam data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def aucCV(features,labels):
    # model = GaussianNB()
    model = make_pipeline(KNNImputer(missing_values=-1, n_neighbors=5),
                          StandardScaler(),
                          GaussianNB())
    scores = cross_val_score(model,features,labels,cv=10,scoring='roc_auc')
    
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    # model = GaussianNB()
    model = make_pipeline(KNNImputer(missing_values=-1, n_neighbors=5),
                          StandardScaler(),
                          GaussianNB())
    model.fit(trainFeatures,trainLabels)
    
    # Use predict_proba() rather than predict() to use probabilities rather
    # than estimated class labels as outputs
    testOutputs = model.predict_proba(testFeatures)[:,1]
    
    return testOutputs
    
# Run this code only if being used as a script, not being imported
if __name__ == "__main__":
    seed = 42
    data1 = np.loadtxt('spamTrain1.csv',delimiter=',')
    data2 = np.loadtxt('spamTrain2.csv',delimiter=',')
    data = np.r_[data1, data2]
    # Separate labels (last column)
    features = data[:,:-1]
    labels = data[:,-1]

    # Evaluating classifier accuracy using 10-fold cross-validation on combined data
    print("10-fold cross-validation mean AUC: ",
          np.mean(aucCV(features,labels)))

    # Create an 80/20 train/test split with fixed seed (do not hardcode counts)
    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(
        features, labels, test_size=0.2, random_state=seed, stratify=labels
    )

    testOutputs = predictTest(trainFeatures,trainLabels,testFeatures)
    print("Test set AUC: ", roc_auc_score(testLabels,testOutputs))

    # Examine outputs compared to labels
    sortIndex = np.argsort(testLabels)
    nTestExamples = testLabels.size
    plt.subplot(2,1,1)
    plt.plot(np.arange(nTestExamples),testLabels[sortIndex],'b.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Target')
    plt.subplot(2,1,2)
    plt.plot(np.arange(nTestExamples),testOutputs[sortIndex],'r.')
    plt.xlabel('Sorted example number')
    plt.ylabel('Output (predicted target)')
    plt.tight_layout()
    plt.show()
    