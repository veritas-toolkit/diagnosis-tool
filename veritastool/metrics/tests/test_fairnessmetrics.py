import pickle
from veritastool.model.model_container import ModelContainer
from veritastool.fairness.credit_scoring import CreditScoring
from veritastool.metrics.performance_metrics import PerformanceMetrics
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.fairness.fairness import Fairness
import numpy as np
import pandas as pd
import pytest
from veritastool.util.errors import *
import sys
sys.path.append("veritastool/examples/customer_marketing_example")
import selection, uplift, util

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
#model_object = LogisticRegression(C=0.1)
model_name = "credit scoring"
model_type = "credit"
y_prob = cs["y_prob"]

#rejection inference
num_applicants = {'SEX': [3500.0, 5000.0], 'MARRIAGE':[3500.0, 5000.0]}
base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}


#Create Model Container and Use Case Object
container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                           fair_priority = "benefit", fair_impact = "significant", 
#                            fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                          num_applicants =num_applicants,  base_default_rate=base_default_rate)
# cre_sco_obj.k = 1

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
import sys

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

cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 85.4, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "expected_profit", revenue = PROFIT_RESPOND, treatment_cost =COST_TREATMENT
#                                   , fairness_metric_value_input = {'isforeign':{'rejected_harm': 0.2} }
                                 )
# cm_uplift_obj.k = 1

def test_execute_all_fair():
    # cre_sco_obj._compute_fairness(1)
    cre_sco_obj.evaluate()
    assert cre_sco_obj.fair_metric_obj.result != None
    # cm_uplift_obj._compute_fairness(1)
    cm_uplift_obj.evaluate()
    assert cm_uplift_obj.fair_metric_obj.result != None

def test_translate_metric():
    cre_sco_obj.feature_importance()
    assert cre_sco_obj.feature_imp_values != None
    cm_uplift_obj.feature_importance()
    assert cm_uplift_obj.feature_imp_values != None

def test_compute_wape_parity():
    cre_sco_obj.curr_p_var = 'SEX'
    cre_sco_obj.y_true = [[],[]]
    cre_sco_obj.y_true[0] = np.array(y_true)
    cre_sco_obj.y_true[1] = np.array(y_true)
    cre_sco_obj.y_pred = [[],[]]
    cre_sco_obj.y_pred[0] = np.array(y_pred)
    cre_sco_obj.y_pred[1] = np.array(y_pred)
    result = FairnessMetrics._compute_wape_parity(cre_sco_obj)
    assert result == (0.216670530952933, 0.4008810572687225)
    
def test_compute_mape_parity():
    cre_sco_obj.sample_weight = [[],[]]
    cre_sco_obj.sample_weight[0] = np.array([0.7 for x in range(len(y_pred))])
    cre_sco_obj.sample_weight[1] = np.array([0.7 for x in range(len(y_pred))])
    result = FairnessMetrics._compute_mape_parity(cre_sco_obj)
    assert result == (46294276723477.69, 394254512833780.9)

def test_compute_mape_parity():
    cre_sco_obj.sample_weight = [[],[]]
    np.random.seed(0)

    cre_sco_obj.sample_weight[0] = np.random.random_sample((len(y_pred),))
    cre_sco_obj.sample_weight[1] = np.random.random_sample((len(y_pred),))
    result = FairnessMetrics._compute_mape_parity(cre_sco_obj)
    assert result == (50612541836597.81, 440356453891928.2)
    
def test__compute_rmse_parity():
    result = FairnessMetrics._compute_rmse_parity(cre_sco_obj)
    assert result == (0.16043449543124988, 0.5535316670230406)    

def test__compute_rmse_parity():
    result = FairnessMetrics._compute_rmse_parity(cre_sco_obj)
    assert result == (0.14359862866686257, 0.5529818274559376)

def test_compute_benefit_from_acquiring():
    cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 85.4, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "expected_profit", fair_metric_name = 'acquire_benefit', revenue = PROFIT_RESPOND, treatment_cost =COST_TREATMENT)
    cm_uplift_obj.feature_importance()
    assert round(cm_uplift_obj.feature_imp_values['isforeign']['isforeign'][1],3) == -0.045

def test_compute_disparate_impact_rejection_inference():
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                           fair_priority = "benefit", fair_impact = "significant",  fair_metric_name = 'disparate_impact',
                           # fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                          num_applicants =num_applicants,  base_default_rate=base_default_rate)
    cre_sco_obj.feature_importance()
    assert round(cre_sco_obj.feature_imp_values['SEX']['SEX'][1],3) == 0.192
    
def test_compute_disparate_impact():
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                           fair_priority = "benefit", fair_impact = "significant",  fair_metric_name = 'disparate_impact',
                           # fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                         )
    cre_sco_obj.feature_importance()
    assert round(cre_sco_obj.feature_imp_values['SEX']['SEX'][1],3) == 0.205
