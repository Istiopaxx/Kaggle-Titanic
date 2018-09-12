import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl
import os
import scipy
from Data_pipeline import full_data_pipeline, load_titanic_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier

def logisticRegModel(train_data, train_label, test_data, test_label):
    log_Reg = LogisticRegression()
    log_Reg.fit(train_data, train_label)
    prediction = log_Reg.predict(test_data)
    accuracy = accuracy_score(test_label, prediction)
    print("Logistic Regression accuracy : ", accuracy)


def suppoVecClasModel(train_data, train_label, test_data, test_label):
    sv_classifier = SVC()
    sv_classifier.fit(train_data, train_label)
    prediction = sv_classifier.predict(test_data)
    accuracy = accuracy_score(test_label, prediction)
    print("SVC accuracy : ", accuracy)


def decisionTreeModel(train_data, train_label, test_data, test_label):
    tree_model = DecisionTreeClassifier()
    tree_model.fit(train_data, train_label)
    prediction = tree_model.predict(test_data)
    accuracy = accuracy_score(test_label, prediction)
    print("Decision Tree accuracy : ", accuracy)


def SVCensembleModel(train_data, train_label, test_data, test_label):
    svc_ensemble = BaggingClassifier(base_estimator=SVC(C=0.3), random_state=97)
    svc_ensemble.fit(train_data, train_label)
    prediction = svc_ensemble.predict(test_data)
    accuracy = accuracy_score(test_label, prediction)
    print("SVC bagging accuracy : ", accuracy)









