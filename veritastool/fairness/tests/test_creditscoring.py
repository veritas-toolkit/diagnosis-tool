import pickle
from veritastool.model.model_container import ModelContainer
from veritastool.fairness.credit_scoring import CreditScoring
from veritastool.metrics.performance_metrics import PerformanceMetrics
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.fairness.fairness import Fairness
# from veritastool.custom.LRwrapper import LRwrapper
import numpy as np
import pandas as pd
import pytest
from veritastool.util.errors import *


#Load Credit Scoring Test Data
#PATH = os.path.abspath(os.path.dirname(__file__))
file = "veritastool/resources/data/credit_score_dict.pickle"
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
model_object = cs["model"]
# model_object = LogisticRegression(C=0.1)
model_name = "credit scoring"
model_type = "credit"
y_prob = cs["y_prob"]

#rejection inference
num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}
# model_object = LRwrapper(model_object)


#Create Model Container and Use Case Object
container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                           fair_priority = "benefit", fair_impact = "significant", 
                           fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                          num_applicants =num_applicants,  base_default_rate=base_default_rate)
#cre_sco_obj.k = 1



def test_check_input():
    cre_sco_obj._model_type_to_metric_lookup[cre_sco_obj.model_params[0].model_type] = ('classification', 4, 2)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj._check_input()
    assert toolkit_exit.type == MyError
    cre_sco_obj._model_type_to_metric_lookup[cre_sco_obj.model_params[0].model_type] = ('classification', 2, 1)
    cre_sco_obj.model_params[0].y_pred = None
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj._check_input()
    assert toolkit_exit.type == MyError

def test_check_special_params():
    #Load Credit Scoring Test Data
    #PATH = os.path.abspath(os.path.dirname(__file__))
    file = "veritastool/resources/data/credit_score_dict.pickle"
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
    model_object = cs["model"]
    # model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = 1
    base_default_rate = 2
    # model_object = LRwrapper(model_object)


    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                                   fair_priority = "benefit", fair_impact = "significant", 
                                   fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                                  num_applicants =num_applicants,  base_default_rate=base_default_rate)
    assert toolkit_exit.type == MyError
    # cre_sco_obj.spl_params = {'num_applicants': 1, 'base_default_rate': 2}
    
    
    #Load Credit Scoring Test Data
    #PATH = os.path.abspath(os.path.dirname(__file__))
    file = "veritastool/resources/data/credit_score_dict.pickle"
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
    model_object = cs["model"]
    # model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': ['3500', 5000], 'MARRIAGE':[3500, 5000]}
    base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}
    # model_object = LRwrapper(model_object)


    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                               fair_priority = "benefit", fair_impact = "significant", 
                               fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                              num_applicants =num_applicants,  base_default_rate=base_default_rate)
    assert toolkit_exit.type == MyError
    # cre_sco_obj.spl_params  = {'num_applicants': {'SEX': ['3500', '5000'], 'MARRIAGE': [3500, 5000]},
     # 'base_default_rate': {'SEX': [0.1, 0.05], 'MARRIAGE': [0.1, 0.05]}}
     
    #Load Credit Scoring Test Data
    #PATH = os.path.abspath(os.path.dirname(__file__))
    file = "veritastool/resources/data/credit_score_dict.pickle"
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
    model_object = cs["model"]
    # model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
    base_default_rate = {'SEX': ['0.10',0.05], 'MARRIAGE':[0.10,0.05]}
    # model_object = LRwrapper(model_object)


    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                               fair_priority = "benefit", fair_impact = "significant", 
                               fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                              num_applicants =num_applicants,  base_default_rate=base_default_rate)
    assert toolkit_exit.type == MyError 
    

    #Load Credit Scoring Test Data
    #PATH = os.path.abspath(os.path.dirname(__file__))
    file = "veritastool/resources/data/credit_score_dict.pickle"
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
    model_object = cs["model"]
    # model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': [-3500, 5000], 'MARRIAGE':[3500, 5000]}
    base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}
    # model_object = LRwrapper(model_object)


    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                               fair_priority = "benefit", fair_impact = "significant", 
                               fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                              num_applicants =num_applicants,  base_default_rate=base_default_rate)
    assert toolkit_exit.type == MyError     

    #Load Credit Scoring Test Data
    #PATH = os.path.abspath(os.path.dirname(__file__))
    file = "veritastool/resources/data/credit_score_dict.pickle"
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
    model_object = cs["model"]
    # model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
    base_default_rate = {'SEX': [-0.10,0.05], 'MARRIAGE':[0.10,0.05]}
    # model_object = LRwrapper(model_object)


    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                               fair_priority = "benefit", fair_impact = "significant", 
                               fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                              num_applicants =num_applicants,  base_default_rate=base_default_rate)
    assert toolkit_exit.type == MyError  

    #Load Credit Scoring Test Data
    #PATH = os.path.abspath(os.path.dirname(__file__))
    file = "veritastool/resources/data/credit_score_dict.pickle"
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
    model_object = cs["model"]
    # model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': [3500, 5000, 3500], 'MARRIAGE':[3500, 5000, 3500]}
    base_default_rate = {'SEX': [-0.10,0.05], 'MARRIAGE':[0.10,0.05]}
    # model_object = LRwrapper(model_object)


    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                               fair_priority = "benefit", fair_impact = "significant", 
                               fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                              num_applicants =num_applicants,  base_default_rate=base_default_rate)
    assert toolkit_exit.type == MyError  

    #Load Credit Scoring Test Data
    #PATH = os.path.abspath(os.path.dirname(__file__))
    file = "veritastool/resources/data/credit_score_dict.pickle"
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
    model_object = cs["model"]
    # model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
    base_default_rate = {'SEX': [0.1, 0.05, 0.1], 'MARRIAGE':[0.1, 0.05, 0.1]}
    # model_object = LRwrapper(model_object)


    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                               fair_priority = "benefit", fair_impact = "significant", 
                               fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                              num_applicants =num_applicants,  base_default_rate=base_default_rate)
    assert toolkit_exit.type == MyError  

    #Load Credit Scoring Test Data
    #PATH = os.path.abspath(os.path.dirname(__file__))
    file = "veritastool/resources/data/credit_score_dict.pickle"
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
    model_object = cs["model"]
    # model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
    base_default_rate = {'SEX': [0.001,0.005], 'MARRIAGE':[0.001,0.005]}
    # model_object = LRwrapper(model_object)


    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                               fair_priority = "benefit", fair_impact = "significant", 
                               fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                              num_applicants =num_applicants,  base_default_rate=base_default_rate)
    assert toolkit_exit.type == MyError 
    
    #Load Credit Scoring Test Data
    #PATH = os.path.abspath(os.path.dirname(__file__))
    file = "veritastool/resources/data/credit_score_dict.pickle"
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
    model_object = cs["model"]
    # model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
    base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}
    # model_object = LRwrapper(model_object)


    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                               fair_priority = "benefit", fair_impact = "significant",fair_metric_name = "mi_independence", 
                               fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                              num_applicants =num_applicants,  base_default_rate=base_default_rate)
    assert toolkit_exit.type == MyError  
    
def test_get_confusion_matrix():
    #Load Credit Scoring Test Data
    #PATH = os.path.abspath(os.path.dirname(__file__))
    file = "veritastool/resources/data/credit_score_dict.pickle"
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
    model_object = cs["model"]
    # model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': [3500.0, 5000.0], 'MARRIAGE':[3500.0, 5000.0]}
    base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}
    # model_object = LRwrapper(model_object)


    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                               fair_priority = "benefit", fair_impact = "significant", 
                               fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                              num_applicants =num_applicants,  base_default_rate=base_default_rate)
                              
    result = cre_sco_obj._get_confusion_matrix(None,None,None)
    assert len(result) == 4
    assert result[0] == None

    result = cre_sco_obj._get_confusion_matrix(None,None,0.25,curr_p_var = 'SEX')
    assert len(result) == 8
    assert result[0] == None
    
    cre_sco_obj.spl_params  = {'num_applicants': {'SEX': [3500, 5000], 'MARRIAGE': [3500, 5000]},
 'base_default_rate': {'SEX': [0.1, 0.05], 'MARRIAGE': [0.1, 0.05]}}
    result = cre_sco_obj._get_confusion_matrix(y_true,y_pred, None)
    assert len(result) == 4
    assert result == (507, 61, 539.0, 7393.0)

    cre_sco_obj._rejection_inference_flag = {'SEX': False, 'MARRIAGE': False}
    cre_sco_obj.spl_params  = {'num_applicants': {'SEX': [3500, 5000], 'MARRIAGE': [3500, 5000]},
 'base_default_rate': {'SEX': [0.1, 0.05], 'MARRIAGE': [0.1, 0.05]}}
    result = cre_sco_obj._get_confusion_matrix(y_true=y_true,y_pred=y_pred,sample_weight = np.array([0.7 for x in range(len(y_pred))]),curr_p_var = 'SEX', feature_mask = cre_sco_obj.feature_mask)
    assert len(result) == 8
    assert result == (113.4000000000003,
 18.199999999999992,
 30.79999999999998,
 45.500000000000036,
 241.49999999999852,
 24.499999999999986,
 26.599999999999984,
 24.499999999999986)
