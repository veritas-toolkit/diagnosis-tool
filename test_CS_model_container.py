from sklearn.linear_model import LogisticRegression
import pickle
from veritastool.model_container import ModelContainer
from veritastool.fairness.credit_scoring import CreditScoring
#from veritastool.performance_metrics import PerformanceMetrics
from veritastool.fairness.fairness_metrics import FairnessMetrics
from veritastool.fairness.fairness import Fairness
from veritastool.fairness.fairness import Fairness
from veritastool.custom.LRwrapper import LRwrapper
import numpy as np
import pandas as pd

from veritastool.utility import check_datatype, check_value, check_label


import pytest
import sys
from copy import deepcopy


feature_imp = pd.DataFrame(data = {'features': ['EDUCATION', 'SEX', 'MARRIAGE', 'AGE'], 'values': [0.04, 0.08, 0.03, 0.02]})
#Load Credit Scoring Test Data
file = "veritastool/resources/data/credit_score_dict.pickle"
input_file = open(file, "rb")
cs = pickle.load(input_file)
input_file = open(file, "rb")
cs = pickle.load(input_file)

#Reduce into two classes
cs["X_train"]['MARRIAGE'] = cs["X_train"]['MARRIAGE'].replace([0, 3],1)
cs["X_test"]['MARRIAGE'] = cs["X_test"]['MARRIAGE'].replace([0, 3],1)
#Model Contariner Parameters
y_true = np.array(cs["y_test"])
y_pred = np.array(cs["y_pred"])
y_train = np.array(cs["y_train"])
p_var = ['SEX', 'MARRIAGE']
p_grp = {'SEX': [1], 'MARRIAGE':[1]}
x_train = cs["X_train"]
x_test = cs["X_test"]
model_object = LogisticRegression(C=0.1)
model_name = "credit scoring"
model_type = "default"
y_prob = cs["y_prob"]

#rejection inference
num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}
model_object = LRwrapper(model_object)

m_container = ModelContainer(y_true=y_true, y_train=y_train, p_var=p_var, p_grp=p_grp, x_train=x_train,  x_test=x_test, \
                           model_object=model_object, model_type =model_type, model_name = model_name,  y_pred = y_pred,\
                           y_prob = y_prob, feature_imp = feature_imp)


def test_model_container():
    #Create Model Container and Use Case Object
    container = ModelContainer(y_true=y_true, y_train=y_train, p_var=p_var, p_grp=p_grp, x_train=x_train,  x_test=x_test, \
                               model_object=model_object, model_type =model_type, model_name = model_name,  y_pred = y_pred,\
                               y_prob = y_prob, feature_imp = feature_imp)
    assert container is not None

    #y_true
    pos_label = [[1]]
    neg_label = None
    s_y_true = check_label(np.array(y_true, dtype=int), pos_label, neg_label)[0]
    assert np.array_equal(s_y_true, container.y_true)

    #y_train
    assert np.array_equal(y_train, container.y_train)

    #p_var
    assert p_var == container.p_var

    #p_grp
    assert p_grp == container.p_grp

    #y_pred
    pos_label = [[1]]
    neg_label = None
    s_y_pred = check_label(np.array(y_pred, dtype=int), pos_label, neg_label)[0]
    assert np.array_equal(s_y_pred, container.y_pred)

    if model_name == 'auto':
        self.model_name = model_type
    assert model_name[0:20] == container.model_name

    assert container.err.queue == []
    



def test_check_data_consistency():
    msg =''
    f_container = deepcopy(m_container)

    #set a new label for y_true (ndarray)
    y_true[0] = 10
    f_container.y_true = y_true
    msg += '[length_error]: y_true labels: given length 3, expected length 2 at check_data_consistency()\n'

    #set a new column for feature_imp
    feature_imp['new_column'] = 1
    f_container.feature_imp = feature_imp
    msg += '[length_error]: feature_imp column: given length 3, expected length 2 at check_data_consistency()\n'

    #change row count of protected_features_cols
    #change column count of protected_features_cols
    protected_features_cols = m_container.protected_features_cols
    f_container.protected_features_cols = protected_features_cols.append(protected_features_cols[0:10])
    msg += '[length_error]: protected_features_cols row: given length 7511, expected length 7500 at check_data_consistency()\n'
    protected_features_cols.loc['new_column'] = 1
    msg += '[length_error]: p_var array: given length 2, expected length 3 at check_data_consistency()\n'





    #change row count of x_train
    x_train = m_container.x_train
    f_container.x_train = x_train.append(x_train[0:10]) #assigned again below
    msg += ''

    #change row count of sample weight
    f_container.sample_weight = np.random.choice(10, 7500, replace=True)
    f_container.sample_weight_train = None
    msg += '[length_error]: sample_weight_train: given length None, expected length not None at check_data_consistency()\n'
        #test another scenario that f_container.sample_weight_train has different number of rows



    #change y_prob dtype to non float
    f_container.y_prob = m_container.y_prob.astype(int)
    msg += ''

    #if both x_test and x_train are df, change the no. of columns to not the same
    x_train['new_column'] = 1
    f_container.x_train = x_train
    msg += '[length_error]: x_train column: given length 24, expected length 23 at check_data_consistency()\n'

    #change pos_label size and neg_label size
    f_container.neg_label = [[0],['neg']]
    msg += '[length_error]: neg_label: given length 2, expected length 1 at check_data_consistency()\n'
    f_container.pos_label = [[1],['pos']]
    msg += '[length_error]: pos_label: given length 2, expected length 1 at check_data_consistency()\n'


    #change below
        # y_true should 1 columns
        # y_true, y_pred, sample weight, are in same shape
        #Based on the length of pos_label, if 1, the y_prob will be nx1
        #Based on the length of pos_label, if 2, the y_prob will be nx4


    #catch the err poping out
    with pytest.raises(SystemExit) as toolkit_exit:
        f_container.check_data_consistency()
    assert toolkit_exit.type == SystemExit
    print( toolkit_exit.value.code)
    #assert toolkit_exit.value.code == msg


def test_check_data_consistency_2():
    msg =''
    f_container = deepcopy(m_container)
    #y_pred and y_prob both are none
    f_container.y_pred = None
    f_container.y_prob = None
    msg += '[length_error]: y_pred and y_prob: given length None for both, expected length not both are None at check_data_consistency()\n'
    #catch the err poping out
    with pytest.raises(SystemExit) as toolkit_exit:
        f_container.check_data_consistency()
    assert toolkit_exit.type == SystemExit
    print( toolkit_exit.value.code)
    #assert toolkit_exit.value.code == msg

# def test_check_protected_columns():
#     f_container = deepcopy(m_container)
#     #set both to none
#     f_container.protected_features_cols = None
#     f_container.x_test = None

#     with pytest.raises(SystemExit) as toolkit_exit:
#         f_container.check_protected_columns()
#     assert toolkit_exit.type == SystemExit
#     print( toolkit_exit.value.code)
#     assert toolkit_exit.value.code == '[length_error]: protected_features_cols and x_test: given length None for both, expected length not both are None at check_protected_columns()\n'
