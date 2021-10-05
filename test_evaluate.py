from sklearn.linear_model import LogisticRegression
import pickle
from veritastool.model_container import ModelContainer
from veritastool.fairness.credit_scoring import CreditScoring
from veritastool.fairness.performance_metrics import PerformanceMetrics
from veritastool.fairness.fairness_metrics import FairnessMetrics
from veritastool.fairness.fairness import Fairness
from veritastool.fairness.fairness import Fairness
from veritastool.custom.LRwrapper import LRwrapper
import numpy as np
import pandas as pd
import os

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
model_object = LogisticRegression(C=0.1)
model_name = "credit scoring"
model_type = "default"
y_prob = cs["y_prob"]

#rejection inference
num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}
model_object = LRwrapper(model_object)


#Create Model Container and Use Case Object
container = ModelContainer(y_true=y_true, y_train=y_train, p_var=p_var, p_grp=p_grp, x_train=x_train,  x_test=x_test, \
                           model_object=model_object, model_type =model_type, model_name = model_name,  y_pred = y_pred,\
                           y_prob = y_prob)

cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "balanced_acc", fair_metric_name = "equal_opportunity")
cre_sco_obj.k = 1
cre_sco_obj.evaluate(output = False)
cre_sco_obj.tradeoff()
cre_sco_obj.feature_importance()
cre_sco_obj.compile()
result = cre_sco_obj.perf_metric_obj.result, cre_sco_obj.fair_metric_obj.result

def test_evaluate():

    assert round(result[0]['perf_metric_values']['selection_rate'][0],3) == 0.757
    
def test_tradeoff():

    assert round(cre_sco_obj.tradeoff_obj.result['SEX']['max_perf_point'][0],3) == 0.407
    
def test_feature_importance():

    assert round(cre_sco_obj.feature_imp_values['SEX']['SEX'][0],3) == -0.205
   
def test_artifact():
    
    assert cre_sco_obj.artifact['features']['SEX']['tradeoff']['th_x'].shape == cre_sco_obj.artifact['features']['SEX']['tradeoff']['th_y'].shape
    assert cre_sco_obj.artifact['features']['SEX']['tradeoff']['fair'].shape != cre_sco_obj.artifact['features']['SEX']['tradeoff']['perf'].shape
    assert cre_sco_obj.array_size == cre_sco_obj.artifact['perf_dynamic']['threshold'].shape[0]
    assert cre_sco_obj.array_size == len(cre_sco_obj.artifact['perf_dynamic']['perf'])
    assert cre_sco_obj.array_size == len(cre_sco_obj.artifact['perf_dynamic']['selection_rate'])
