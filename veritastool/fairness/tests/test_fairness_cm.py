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
#cm_uplift_obj.k = 1
cm_uplift_obj.compile()
# cm_uplift_obj.evaluate(visualize=True)
cm_uplift_obj.tradeoff()
cm_uplift_obj.feature_importance()
cm_uplift_obj.compile()
cm_uplift_obj.evaluate()

def test_evaluate():

    assert round(cm_uplift_obj.perf_metric_obj.result['perf_metric_values']['emp_lift'][0],3) == 0.171
   
def test_artifact():
    
    assert cm_uplift_obj.artifact['features']['isforeign']['tradeoff']['th_x'].shape == cm_uplift_obj.artifact['features']['isforeign']['tradeoff']['th_y'].shape
    assert cm_uplift_obj.artifact['features']['isforeign']['tradeoff']['fair'].shape == cm_uplift_obj.artifact['features']['isforeign']['tradeoff']['perf'].shape
    assert cm_uplift_obj.array_size == cm_uplift_obj.artifact['perf_dynamic']['threshold'].shape[0]
    assert cm_uplift_obj.array_size == len(cm_uplift_obj.artifact['perf_dynamic']['perf'])
    assert cm_uplift_obj.array_size == len(cm_uplift_obj.artifact['perf_dynamic']['selection_rate'])
    
def test_fairness_conclusion():
    if cm_uplift_obj.fair_threshold < 1:
        assert cm_uplift_obj.fair_threshold == cm_uplift_obj.fair_conclusion['isforeign']['threshold']
    else:
        value = round((1 - cm_uplift_obj.fair_conclusion['isforeign']['threshold']) *100)
        assert cm_uplift_obj.fair_threshold == 85
    assert cm_uplift_obj.fair_conclusion['isforeign']['fairness_conclusion'] in ('fair','unfair')

def test_compute_fairness():
    if cm_uplift_obj.fairness_metric_value_input is not None :
        assert cm_uplift_obj.fairness_metric_value_input['isforeign']['rejected_harm'] == cm_uplift_obj.fair_metric_obj.result['isforeign']['fair_metric_values']['rejected_harm'][0]
    
def test_fairness_metric_value_input_check():
    cm_uplift_obj.fairness_metric_value_input = {'INCOME': {'fpr_parity': 0.2}}
    cm_uplift_obj._fairness_metric_value_input_check()
    assert cm_uplift_obj.fairness_metric_value_input == None
    
    cm_uplift_obj.fairness_metric_value_input = {'isforeign': {'other_metric': 0.2}}
    cm_uplift_obj._fairness_metric_value_input_check()
    assert cm_uplift_obj.fairness_metric_value_input == None
    
def test_compile():

    assert cm_uplift_obj.evaluate_status == 1
    assert cm_uplift_obj.evaluate_status_cali == False
    assert cm_uplift_obj.evaluate_status_perf_dynamics == True
    assert cm_uplift_obj.tradeoff_status == 1
    assert cm_uplift_obj.feature_imp_status == 1
    assert cm_uplift_obj.feature_imp_status_loo == True
    assert cm_uplift_obj.feature_imp_status_corr == True
    
def test_compile_skip():
    cm_uplift_obj.feature_imp_status = 0
    cm_uplift_obj.tradeoff_status = 0
    cm_uplift_obj.feature_imp_status_corr = False
    cm_uplift_obj.compile(skip_tradeoff_flag=1, skip_feature_imp_flag=1)
    assert cm_uplift_obj.feature_imp_status == -1
    assert cm_uplift_obj.tradeoff_status == -1
    
def test_tradeoff():

    assert round(cm_uplift_obj.tradeoff_obj.result['isforeign']['max_perf_point'][0],3) == 0.272
    cm_uplift_obj.model_params[0].y_prob = None
    cm_uplift_obj.tradeoff()
    assert cm_uplift_obj.tradeoff_status == -1
    cm_uplift_obj.tradeoff_obj.result= None
    cm_uplift_obj.tradeoff()
    assert cm_uplift_obj.tradeoff_status == -1
    
def test_feature_importance():
    cm_uplift_obj.feature_imp_status = 0
    cm_uplift_obj.evaluate_status = 0
    cm_uplift_obj.feature_importance()
    assert round(cm_uplift_obj.feature_imp_values['isforeign']['isforeign'][0],3) == 63332.82
    cm_uplift_obj.feature_imp_status = -1
    cm_uplift_obj.feature_importance()
    assert cm_uplift_obj.feature_imp_values == None
    x_train = np.array([1,2,3])
    x_test = np.array([1,2,3])
    cm_uplift_obj.feature_importance()
    assert isinstance(x_train, pd.DataFrame) == False
    assert isinstance(x_train, pd.DataFrame) == False
    
    
# def test_e_lift():
    # result = cm_uplift_obj._get_e_lift()
    # assert result == None

def test_feature_mask():
    assert len(cm_uplift_obj.model_params[0].x_test) == len(cm_uplift_obj.feature_mask['isforeign'])     

def test_base_input_check():
    cm_uplift_obj.fair_metric_name = 'mi_independence'
    cm_uplift_obj.fair_threshold = 43
    cm_uplift_obj.fairness_metric_value_input = {'isforeign': {'other_metric': 0.2}}
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._base_input_check()
    assert toolkit_exit.type == MyError
    
def test_model_type_input():
    cm_uplift_obj.model_params[0].model_type = 'svm'
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._model_type_input()
    assert toolkit_exit.type == MyError
    
    cm_uplift_obj._model_type_to_metric_lookup[cm_uplift_obj.model_params[0].model_type] = ('uplift', 4, 4)
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._model_type_input()
    assert toolkit_exit.type == MyError

    cm_uplift_obj.model_params[0].model_type = 'uplift'
    cm_uplift_obj.model_params[1].model_name = 'duplicate'
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._model_type_input()
    assert toolkit_exit.type == MyError
    
def test_fairness_tree():
    cm_uplift_obj.fair_impact = 'normal'
    cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj.fair_metric_name == 'equal_opportunity'
    cm_uplift_obj.fair_concern = 'inclusive'
    cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj.fair_metric_name == 'fpr_parity'
    cm_uplift_obj.fair_concern = 'both'
    cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj.fair_metric_name == 'equal_odds'
    cm_uplift_obj.fair_impact = 'selective'
    cm_uplift_obj.fair_concern = 'eligible'
    cm_uplift_obj.fair_priority = 'benefit'
    cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj.fair_metric_name == 'ppv_parity'
    cm_uplift_obj.fair_impact = 'selective'
    cm_uplift_obj.fair_concern = 'inclusive'
    cm_uplift_obj.fair_priority = 'benefit'
    cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj.fair_metric_name == 'fdr_parity'
    cm_uplift_obj.fair_concern = 'both'
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._fairness_tree()
    assert toolkit_exit.type == MyError
    cm_uplift_obj.fair_impact = 'normal'
    cm_uplift_obj.fair_concern = 'inclusive'
    cm_uplift_obj.fair_priority = 'harm'
    cm_uplift_obj._fairness_tree()
    assert cm_uplift_obj.fair_metric_name == 'fpr_parity'
    
    cm_uplift_obj.fair_concern = 'eligible'
    cm_uplift_obj.fair_priority = 'benefit'
    cm_uplift_obj.fair_impact = 'normal'
    cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj.fair_metric_name == 'tnr_parity'
    cm_uplift_obj.fair_concern = 'inclusive'
    cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj.fair_metric_name == 'fnr_parity'
    cm_uplift_obj.fair_concern = 'both'
    cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj.fair_metric_name == 'neg_equal_odds'
    cm_uplift_obj.fair_impact = 'selective'
    cm_uplift_obj.fair_concern = 'eligible'
    cm_uplift_obj.fair_priority = 'benefit'
    cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj.fair_metric_name == 'npv_parity'
    cm_uplift_obj.fair_impact = 'selective'
    cm_uplift_obj.fair_concern = 'inclusive'
    cm_uplift_obj.fair_priority = 'benefit'
    cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj.fair_metric_name == 'for_parity'
    cm_uplift_obj.fair_concern = 'both'
    with pytest.raises(MyError) as toolkit_exit:
        cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert toolkit_exit.type == MyError
    cm_uplift_obj.fair_impact = 'normal'
    cm_uplift_obj.fair_concern = 'inclusive'
    cm_uplift_obj.fair_priority = 'harm'
    cm_uplift_obj._fairness_tree(is_pos_label_favourable = False)
    assert cm_uplift_obj.fair_metric_name == 'fnr_parity'
