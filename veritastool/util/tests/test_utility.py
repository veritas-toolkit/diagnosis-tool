import math
import pickle
import numpy as np
import pandas as pd
import pytest
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from veritastool.model.model_container import ModelContainer
from veritastool.util.utility import *
#from veritastool.util.utility import check_datatype, check_value, convert_to_set, check_label, get_cpu_count, check_multiprocessing
from veritastool.util.errors import MyError

#sample feature_imp
feature_imp = pd.DataFrame(data = {'features': ['EDUCATION', 'SEX', 'MARRIAGE', 'AGE'], 'values': [0.04, 0.08, 0.03, 0.02]})

#Load Credit Scoring Test Data
#file = r"C:\Users\brian.zheng\OneDrive - Accenture\General\05 Deliverables\T2\test_credit_score_dict.pickle"
file = "veritastool/examples/data/credit_score_dict.pickle"
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
model_type = "credit"
y_prob = cs["y_prob"]

#rejection inference
num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}

container = ModelContainer(y_true=y_true, y_train=y_train, p_var=p_var, p_grp=p_grp, x_train=x_train,  x_test=x_test, \
                           model_object=model_object, model_type =model_type, model_name = model_name,  y_pred = y_pred,\
                           y_prob = y_prob, feature_imp = feature_imp)

def test_check_datatype():
    f_container = deepcopy(container)
    msg = ''

    f_container.y_true = None
    msg += '[type_error]: y_true: given <class \'NoneType\'>, expected [<class \'list\'>, <class \'numpy.ndarray\'>, <class \'pandas.core.series.Series\'>] at check_datatype()\n'
    
    f_container.y_pred = tuple(f_container.y_pred)
    msg += '[type_error]: y_pred: given <class \'tuple\'>, expected [<class \'NoneType\'>, <class \'list\'>, <class \'numpy.ndarray\'>, <class \'pandas.core.series.Series\'>] at check_datatype()\n'
    
    f_container.p_grp = {'SEX': np.array([1]), 'MARRIAGE':[1]}
    msg += '[type_error]: p_grp values: given <class \'numpy.ndarray\'>, expected list at check_datatype()\n'

    # f_container._input_validation_lookup['sample_weight_train'] = [None, (0, np.inf)]
    # msg += '[type_error]: sample_weight_train: given <class \'str\'>, expected None at check_datatype()\n'

    f_container._input_validation_lookup['new_variable'] = [(list,), str]
    msg += '[type_error]: new_variable: given None, expected [<class \'list\'>] at check_datatype()\n'

    f_container._input_validation_lookup['new_variable2'] = [None, str]
    
    #catch the err poping out
    with pytest.raises(Exception) as toolkit_exit:
        check_datatype(f_container)
    # print(toolkit_error.type)
    assert toolkit_exit.type == MyError
    # print( toolkit_error.value.message)
    assert toolkit_exit.value.message == msg

def test_check_value():
    f_container = deepcopy(container)
    msg = ''
    
    #when the length is 1, check_value will not be performed for this variable
    f_container.new_variable2 = 'random_var'
    f_container._input_validation_lookup['new_variable2'] = [str, ]

    #change y_prob
    f_container.y_prob[0] = 10
    msg += '[value_error]: y_prob: given range [0.004324126464885938 : 10.0] , expected (-0.01, 1.01) at check_value()\n'

    #change p_var
    f_container.p_var = ['SEX', 123]
    msg += '[value_error]: p_var: given <class \'int\'>, expected <class \'str\'> at check_value()\n'

    #change protected_features_cols columns
    f_container.protected_features_cols['new_column'] = 1
    msg += '[column_value_error]: protected_features_cols: given [\'MARRIAGE\', \'SEX\'] expected [\'MARRIAGE\', \'SEX\', \'new_column\'] at check_value()\n'

    #change p_grp
    f_container.p_grp = {'SEX': [1], 'MARRIAGE':[1], 'RELIGION': [1]}
    msg += '[value_error]: p_grp: given [\'MARRIAGE\', \'RELIGION\', \'SEX\'], expected [\'MARRIAGE\', \'SEX\'] at check_value()\n'

    #change feature_imp
    f_container.feature_imp['new_column'] = 1
    msg += '[length_error]: feature_imp: given length 3, expected length 2 at check_value()\n'

    #change model type
    f_container.model_type = 'random_type'
    msg += '[value_error]: model_type: given random_type, expected [\'credit\', \'propensity\', \'rejection\', \'uplift\'] at check_value()\n'

    #change pos_label
    f_container.pos_label = [[1, 'pos']]
    msg += '[value_error]: pos_label: given [\'1\', \'pos\'], expected [0, 1] at check_value()\n'

    #catch the err poping out
    with pytest.raises(Exception) as toolkit_exit:
        check_value(f_container)
    assert toolkit_exit.type == MyError
    # print('====== test_check_value() =======\n')
    # print(toolkit_exit.value.message)
    # print('====== test_check_value() expected msg =======\n')
    # print(msg)
    assert toolkit_exit.value.message == msg

    f_container2 = deepcopy(container)
    msg = ''
    #f_container2

    #change _input_validation_lookup, remove 2nd element to skip check value
    f_container2._input_validation_lookup['y_prob'] = [(list, np.ndarray, pd.Series, pd.DataFrame), (-0.01,1.01,10)]
    f_container2._input_validation_lookup['p_var'] =  [(list,), ]

    #change p_var
    f_container2.p_grp =  {'SEX': [1,2,3], 'MARRIAGE':[1]}
    msg += '[value_error]: p_grp SEX: given [1, 2, 3], expected [1, 2] at check_value()\n'

    #change feature_imp
    f_container2.feature_imp = pd.DataFrame(data = {'features': ['EDUCATION', 'SEX', 'MARRIAGE', 'AGE'], 'values': ['important', 0.08, 0.03, 0.02]})
    msg += '[column_value_error]: feature_imp: given object expected float64 at check_value()\n'
    
    f_container2.fair_neutral_tolerance = 0.1
    f_container2._input_validation_lookup['fair_neutral_tolerance'] = [(int, float), (0, 0.01)]
    msg += '[value_error]: fair_neutral_tolerance: given 0.1, expected (0, 0.01) at check_value()\n'

    f_container2.new_var = 'new_variable'
    f_container2._input_validation_lookup['new_var'] = [(str,), (0, 1)]
    msg += '[value_error]: new_var: given a range of (0, 1), expected a range for <class \'str\'> at check_value()\n'

    #catch the err poping out
    with pytest.raises(Exception) as toolkit_exit:
        check_value(f_container2)
    assert toolkit_exit.type == MyError
    # print('====== test_check_value() =======\n')
    # print(toolkit_exit.value.message)
    # print('====== test_check_value()2 expected msg =======\n')
    # print(msg)
    assert toolkit_exit.value.message == msg


def test_convert_to_set():
    assert convert_to_set('s') == {'s',}
    assert convert_to_set(1) =={1,}
    assert convert_to_set({1,2,3}) == {1,2,3}
    assert convert_to_set((1,2)) == {1,2}
    a = [1,2,3,4,5,5,5]
    assert convert_to_set(a) == {1,2,3,4,5}

def test_check_label():
    #file_prop = r"C:\Users\brian.zheng\OneDrive - Accenture\Desktop\BZ Veritas Code\1007 Pytest\veritas\test\utility_test_data.pickle"
    file_prop = "veritastool/examples/data/test_utility_data.pickle"
    input_prop = open(file_prop, "rb")
    cm_prop = pickle.load(input_prop)
    y_true_prop = cm_prop["y_true_prop"]
    y_true_new, pos_label2 = check_label(y=y_true_prop, pos_label=[['TR'], ['CR']], neg_label=[['TN'], ['CN']])
    labels, counts = np.unique(y_true_new, return_counts=True)
    assert np.array_equal(labels, np.array(['CN', 'CR', 'TN', 'TR']))
    assert np.array_equal(counts, np.array([3734, 2277, 2476, 1513]))
    
    y = np.array(['XR', 'CN', 'CR', 'TN', 'TR', 'XR', 'CN', 'CR', 'TN', 'TR'])
    msg = '[conflict_error]: pos_label, neg_label: inconsistent values [[\'TR\'], [\'CR\'], [\'TN\'], [\'CN\']] at check_label()\n'
    #catch the err poping out
    with pytest.raises(Exception) as toolkit_exit:
        y_new, pos_label2 = check_label(y=y, pos_label=[['TR'], ['CR']], neg_label=[['TN'], ['CN']])
    assert toolkit_exit.type == MyError
    # print( toolkit_exit.value.message)
    assert toolkit_exit.value.message == msg

    y = np.array([1,1,1,1,1,1,1])
    msg = '[value_error]: pos_label: given [1], expected not all y_true labels at check_label()\n'
    #catch the err poping out
    with pytest.raises(Exception) as toolkit_exit:
        y_new, pos_label2 = check_label(y=y, pos_label=[[1]], neg_label=[[0]])
    assert toolkit_exit.type == MyError
    # # print('====== test_check_label() =======\n')
    # # print(toolkit_exit.value.message)
    # # print('====== test_check_label() expected msg =======\n')
    # # print(msg)
    assert toolkit_exit.value.message == msg
    
    y = np.array([0,0,0,0,0,0])
    msg = '[value_error]: pos_label: given [1], expected {0} at check_label()\n'
    #catch the err poping out
    with pytest.raises(Exception) as toolkit_exit:
        y_new, pos_label2 = check_label(y=y, pos_label=[[1]], neg_label=[[0]])
    assert toolkit_exit.type == MyError
    # # print('====== test_check_label() =======\n')
    # # print(toolkit_exit.value.message)
    # # print('====== test_check_label() expected msg =======\n')
    # # print(msg)
    assert toolkit_exit.value.message == msg

def test_get_cpu_count():
    assert get_cpu_count() > 0

def test_check_multiprocessing():
    cpu_count = math.floor(get_cpu_count()/2)
    assert check_multiprocessing(-1) == 1
    assert check_multiprocessing(0) == cpu_count
    assert check_multiprocessing(1) == 1
    assert check_multiprocessing(2) == min(cpu_count, 2)
    assert check_multiprocessing(8) == min(cpu_count, 8)
    assert check_multiprocessing(32) == min(cpu_count, 32)

def test_test_function_cs():
    test_function_cs()

# def test_test_function_cm():
#     test_function_cm()
