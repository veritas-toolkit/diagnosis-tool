import pickle
import numpy as np
import pandas as pd
#from phase1_functions import expected_profit, expected_reject_harm
from veritastool.model.model_container import ModelContainer
from veritastool.fairness.customer_marketing import CustomerMarketing
from veritastool.metrics.performance_metrics import PerformanceMetrics
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.fairness.fairness import Fairness
import pytest
from veritastool.util.errors import *
import sys
sys.path.append("veritastool/examples/customer_marketing_example")
import selection, uplift, util

#Load Credit Scoring Test Data
#PATH = os.path.abspath(os.path.dirname(__file__)))
file_prop = "veritastool/resources/data/mktg_uplift_acq_dict.pickle"
file_rej = "veritastool/resources/data/mktg_uplift_rej_dict.pickle"
input_prop = open(file_prop, "rb")
input_rej = open(file_rej, "rb")
cm_prop = pickle.load(input_prop)
cm_rej = pickle.load(input_rej)

#Model Container Parameters
#Rejection Model
y_true_rej = cm_rej["y_test"]
y_pred_rej = cm_rej["y_test"]
y_train_rej = cm_rej["y_train"]
p_var_rej = ['isforeign', 'isfemale']
p_grp_rej = {'isforeign':[0], 'isfemale':[0]}
x_train_rej = cm_rej["X_train"].drop(['ID'], axis = 1)
x_test_rej = cm_rej["X_test"].drop(['ID'], axis = 1)
model_object_rej = cm_rej['model']
model_name_rej = "cm_rejection"
model_type_rej = "uplift"
y_prob_rej = cm_rej["y_prob"]
# y_prob_rej = None
data = {"FEATURE" :['income', 'noproducts', 'didrespond', 'age', 'isfemale',
       'isforeign'], 
        "VALUE":[0.3, 0.2, 0.15, 0.1, 0.05, 0.03]}
feature_importance_prop = pd.DataFrame(data)

#Propensity Model
y_true_prop = cm_prop["y_test"]
y_pred_prop = cm_prop["y_test"]
y_train_prop = cm_prop["y_train"]
p_var_prop = ['isforeign', 'isfemale']
p_grp_prop = {'isforeign':[0], 'isfemale':[0]}
x_train_prop = cm_prop["X_train"].drop(['ID'], axis = 1)
x_test_prop = cm_prop["X_test"].drop(['ID'], axis = 1)
model_object_prop = cm_prop['model']
model_name_prop = "cm_propensity" 
model_type_prop = "uplift"
y_prob_prop = cm_prop["y_prob"]
#y_prob_prop = None
data = {"FEATURE" :['income', 'noproducts', 'didrespond', 'age', 'isfemale',
       'isforeign'], 
        "VALUE":[0.3, 0.2, 0.15, 0.1, 0.05, 0.03]}
feature_importance_rej = pd.DataFrame(data)

PROFIT_RESPOND = 190
COST_TREATMENT =20

container_rej = ModelContainer(y_true = y_true_rej, y_pred = y_true_rej, y_prob = y_prob_rej, y_train= y_train_rej, p_var = p_var_rej, p_grp = p_grp_rej, x_train = x_train_rej,  x_test = x_test_rej, model_object = model_object_rej,  model_name = model_name_rej, model_type = model_type_rej,  pos_label=[['TR'], ['CR']], neg_label=[['TN'], ['CN']], predict_op_name = "predict_proba",feature_imp=feature_importance_rej)
container_prop = container_rej.clone(y_true = y_true_prop,  y_train = y_train_prop, model_object = model_object_prop, y_pred=None, y_prob=y_prob_prop, train_op_name="fit",
             predict_op_name ="predict_proba", feature_imp=None, sample_weight=None, pos_label=[['TR'], ['CR']], neg_label=[['TN'], ['CN']])

cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 85.4, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "expected_profit", revenue = PROFIT_RESPOND, treatment_cost =COST_TREATMENT, fairness_metric_value_input = {'isforeign':{'rejected_harm': 0.2} })
# cm_uplift_obj.k = 1

def test_check_input():
    cm_uplift_obj._model_type_to_metric_lookup[cm_uplift_obj.model_params[0].model_type] = ('uplift', 4, 4)
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._check_input()
    assert toolkit_exit.type == MyError
    
    cm_uplift_obj._model_type_to_metric_lookup[cm_uplift_obj.model_params[0].model_type]  = ('uplift', 4, 2)
    cm_uplift_obj.model_params[0].y_prob = None
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._check_input()
    assert toolkit_exit.type == MyError
    
    cm_uplift_obj.model_params[0].model_type = 'rejection'
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._check_input()
    assert toolkit_exit.type == MyError

    cm_uplift_obj.model_params[0].model_type = 'uplift'
    cm_uplift_obj.spl_params = {'revenue': '190', 'treatment_cost': 20}
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._check_input()
    assert toolkit_exit.type == MyError

    cm_uplift_obj.spl_params = {'revenue': -190, 'treatment_cost': 20}
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._check_input()
    assert toolkit_exit.type == MyError

    cm_uplift_obj.spl_params = {'revenue': 10, 'treatment_cost': 20}
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._check_input()
    assert toolkit_exit.type == MyError

def test_get_confusion_matrix():
    result = cm_uplift_obj._get_confusion_matrix(None,None,None)
    assert len(result) == 4
    assert result[0] == None

    #Load Credit Scoring Test Data
    #file = r"C:\Users\m.bin.kamaluddin\Accenture\MAS veritastool Toolkit - Documents\General\05 Deliverables\T2\credit_score_dict.pickle"
    file = "veritastool/resources/data/credit_score_dict.pickle"

    input_file = open(file, "rb")
    cs = pickle.load(input_file)

    #Reduce into two classes
    cs["X_train"]['MARRIAGE'] = cs["X_train"]['MARRIAGE'].replace([0, 3],1)
    cs["X_test"]['MARRIAGE'] = cs["X_test"]['MARRIAGE'].replace([0, 3],1)
    #Model Contariner Parameters
    y_true = np.array(cs["y_test"])
    y_pred = np.array(cs["y_pred"])
    # cm_uplift_obj._model_type_to_metric_lookup[cm_uplift_obj.model_params[0].model_type] = ('classification', 2, 1)
    # result = cm_uplift_obj._get_confusion_matrix(y_true,y_pred, None)
    # assert result == (507, 61, 82, 100)

    result = cm_uplift_obj._get_confusion_matrix(y_true,None,None)
    assert len(result) == 4
    assert result[0] == None
       
    # cm_uplift_obj._model_type_to_metric_lookup[cm_uplift_obj.model_params[0].model_type] = ('classification', 2, 1)
    # result = cm_uplift_obj._get_confusion_matrix(y_true,y_pred, sample_weight = np.array([0.7 for x in range(len(y_pred))]))
    # assert result == (354.8999999999967, 42.700000000000024, 57.400000000000084, 70.00000000000013)
    
    result = cm_uplift_obj._get_confusion_matrix(y_true,None,None,curr_p_var = 'isforeign')
    assert len(result) == 8
    assert result[0] == None
    
def test_select_fairness_metric_name():
    cm_uplift_obj.fair_metric_name = 'auto'
    cm_uplift_obj.model_params[0].model_type = 'propensity'
    cm_uplift_obj._select_fairness_metric_name()
    assert cm_uplift_obj.fair_metric_name == 'ppv_parity'

    cm_uplift_obj.fair_metric_name = 'disparate_impact'
    cm_uplift_obj._select_fairness_metric_name()
    assert cm_uplift_obj.fair_metric_name == 'disparate_impact'

    cm_uplift_obj.fair_metric_name = 'auto'
    cm_uplift_obj.model_params[0].model_type = 'classification'
    cm_uplift_obj._select_fairness_metric_name()
    assert cm_uplift_obj.fair_metric_name == 'npv_parity'

    # cm_uplift_obj.model_params[0].model_type = 'uplift'
    # cm_uplift_obj.fair_metric_name = 'mi_independence'
    # with pytest.raises(MyError) as toolkit_exit:
        # cm_uplift_obj._select_fairness_metric_name()
    # assert toolkit_exit.type == MyError 

def test_get_e_lift():
    cm_uplift_obj.model_params[0].model_type = 'classification'
    result = cm_uplift_obj._get_e_lift()
    assert result == None

def test_compute_pred_outcome():
    cm_uplift_obj.model_params[0].model_type = 'classification'
    result = cm_uplift_obj._compute_pred_outcome(y_pred_new=None)
    assert result == None

    cm_uplift_obj.model_params[0].model_type = 'uplift'
    result = cm_uplift_obj._compute_pred_outcome(y_pred_new=[None,None])
    assert result == None
