import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import os
import csv
from Data import load_titanic_data, full_data_pipeline
from Model import logisticRegModel, suppoVecClasModel, decisionTreeModel, SVCensembleModel


data = load_titanic_data()
train_data, train_label, test_data, test_label = full_data_pipeline(data)


logisticRegModel(train_data, train_label, test_data, test_label)
suppoVecClasModel(train_data, train_label, test_data, test_label)
decisionTreeModel(train_data, train_label, test_data, test_label)
SVCensembleModel(train_data, train_label, test_data, test_label)



