import pickle
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)
from veritastool.model.model_container import ModelContainer
from veritastool.usecases.credit_scoring import CreditScoring
from veritastool.metrics.performance_metrics import PerformanceMetrics
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.principles.fairness import Fairness
from veritastool.model.modelwrapper import ModelWrapper
from veritastool.metrics.newmetric import NewMetric
import numpy as np
import pandas as pd
import pytest
from veritastool.util.errors import *
from sklearn.linear_model import LogisticRegression
  
def test_newmetric():
    #Load Credit Scoring Test Data
    file = os.path.join(project_root, 'veritastool', 'examples', 'data', 'credit_score_dict.pickle')
    input_file = open(file, "rb")
    cs = pickle.load(input_file)

    #Reduce into two classes
    cs["X_train"]['MARRIAGE'] = cs["X_train"]['MARRIAGE'].replace([0, 3],1)
    cs["X_test"]['MARRIAGE'] = cs["X_test"]['MARRIAGE'].replace([0, 3],1)
    #Model Contariner Parameters
    y_true = np.array(cs["y_test"])
    y_pred = np.array(cs["y_pred"])
    y_train = np.array(cs["y_train"])
    p_grp = {'SEX': [1], 'MARRIAGE':[1]}
    up_grp = {'SEX': [2], 'MARRIAGE':[2]}
    x_train = cs["X_train"]
    x_test = cs["X_test"]
    model_name = "credit_scoring"
    model_type = "classification"
    y_prob = cs["y_prob"]
    model_obj = LogisticRegression(C=0.1)
    model_obj.fit(x_train, y_train)
    
    test = NewMetric()
    result = test.compute()
    assert result == (0,0) 
