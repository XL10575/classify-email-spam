# -*- coding: utf-8 -*-
"""
Demo of 10-fold cross-validation using Gaussian naive Bayes on spam data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone

def _candidate_pipelines(random_state):
    """Build a small set of strong candidate pipelines.

    All preprocessing is inside the pipeline to avoid data leakage.
    """
    candidates = []

    # 1) Logistic Regression (probabilistic, usually strong on AUC)
    for C in [0.3, 1.0, 3.0]:
        pipe = Pipeline([
            ("imputer", SimpleImputer(missing_values=-1, strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                       max_iter=2000, random_state=random_state))
        ])
        candidates.append((f"logreg_C{C}", pipe))

    # 2) Random Forest (robust, handles non-linearities)
    for n_estimators in [200, 400]:
        for max_depth in [None, 10]:
            pipe = Pipeline([
                ("imputer", SimpleImputer(missing_values=-1, strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    random_state=random_state,
                ))
            ])
            candidates.append((f"rf_{n_estimators}_d{max_depth}", pipe))

    # 3) Gradient Boosting (often good on tabular data)
    for max_depth in [2, 3]:
        pipe = Pipeline([
            ("imputer", SimpleImputer(missing_values=-1, strategy="median")),
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=max_depth,
                random_state=random_state,
            ))
        ])
        candidates.append((f"gb_dt{max_depth}", pipe))

    # 4) Improved Naive Bayes variant (baseline family)
    pipe_gnb = Pipeline([
        ("imputer", KNNImputer(missing_values=-1, n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("clf", GaussianNB()),
    ])
    candidates.append(("gnb_knnimp", pipe_gnb))

    return candidates


def _select_best_model(features, labels, random_state=42, n_splits=5):
    """Select the best pipeline by mean ROC AUC using stratified CV."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_name = None
    best_score = -np.inf
    best_pipeline = None

    for name, pipeline in _candidate_pipelines(random_state):
        scores = cross_val_score(pipeline, features, labels, cv=cv, scoring="roc_auc")
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_name = name
            best_pipeline = clone(pipeline)

    return best_name, best_score, best_pipeline


def aucCV(features,labels):
    name, score, pipeline = _select_best_model(features, labels, random_state=42, n_splits=10)
    # Recompute fold scores for the chosen pipeline to report distribution
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, features, labels, cv=cv, scoring='roc_auc')
    return scores

def predictTest(trainFeatures,trainLabels,testFeatures):
    # Model selection strictly on training set to avoid leakage
    _, _, model = _select_best_model(trainFeatures, trainLabels, random_state=42, n_splits=5)
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
    