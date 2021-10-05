import pickle
import numpy as np
import pandas as pd
#from phase1_functions import expected_profit, expected_reject_harm
from veritastool.model_container import ModelContainer
from veritastool.fairness.customer_marketing import CustomerMarketing
from veritastool.fairness.performance_metrics import PerformanceMetrics
from veritastool.fairness.fairness_metrics import FairnessMetrics
from veritastool.fairness.fairness import Fairness


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

container_rej = ModelContainer(y_true = y_true_rej, y_pred = y_true_rej, y_prob = y_prob_rej, y_train= y_train_rej, p_var = p_var_rej, p_grp = p_grp_rej, x_train = x_train_rej,  x_test = x_test_rej, model_object = model_object_rej,  model_name = model_name_rej, model_type = model_type_rej,  pos_label=[['TR'], ['CR']], neg_label=[['TN'], ['CN']], predict_op_name = "predict_proba", feature_imp = feature_importance_rej)
container_prop = container_rej.clone(y_true = y_true_prop,  y_train = y_train_prop, model_object = model_object_prop, y_pred=None, y_prob=y_prob_prop, train_op_name="fit",
             predict_op_name ="predict_proba", feature_imp=None, sample_weight=None, pos_label=[['TR'], ['CR']], neg_label=[['TN'], ['CN']])

cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 20.4, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "expected_profit", revenue = PROFIT_RESPOND, treatment_cost =COST_TREATMENT)
cm_uplift_obj.k = 1
cm_uplift_obj.evaluate(output = False)
cm_uplift_obj.tradeoff()
cm_uplift_obj.feature_importance()
cm_uplift_obj.compile()

def test_evaluate():

    assert round(cm_uplift_obj.perf_metric_obj.result['perf_metric_values']['emp_lift'][0],3) == 0.171
    
def test_tradeoff():

    assert round(cm_uplift_obj.tradeoff_obj.result['isforeign']['max_perf_point'][0],3) == -0.3
    
def test_feature_importance():

    assert round(cm_uplift_obj.feature_imp_values['isforeign']['isforeign'][0],3) == 63332.82
   
def test_artifact():
    
    assert cm_uplift_obj.artifact['features']['isforeign']['tradeoff']['th_x'].shape == cm_uplift_obj.artifact['features']['isforeign']['tradeoff']['th_y'].shape
    assert cm_uplift_obj.artifact['features']['isforeign']['tradeoff']['fair'].shape == cm_uplift_obj.artifact['features']['isforeign']['tradeoff']['perf'].shape
    assert cm_uplift_obj.array_size == cm_uplift_obj.artifact['perf_dynamic']['threshold'].shape[0]
    assert cm_uplift_obj.array_size == len(cm_uplift_obj.artifact['perf_dynamic']['perf'])
    assert cm_uplift_obj.array_size == len(cm_uplift_obj.artifact['perf_dynamic']['selection_rate'])
