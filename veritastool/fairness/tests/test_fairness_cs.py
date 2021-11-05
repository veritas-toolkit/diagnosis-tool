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
container = ModelContainer(y_true=y_true, y_train=y_train, p_var=p_var, p_grp=p_grp, x_train=x_train,  x_test=x_test, \
                           model_object=model_object, model_type =model_type, model_name = model_name,  y_pred = y_pred,\
                           y_prob = y_prob)

cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 43.2, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "balanced_acc", fair_metric_name = "disparate_impact",fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} })
#cre_sco_obj.k = 1
# cre_sco_obj.fair_metric_name = 'disparate_impact'
# cre_sco_obj.compile()
# result = cre_sco_obj.perf_metric_obj.result, cre_sco_obj.fair_metric_obj.result

cre_sco_obj.compile()
# cre_sco_obj.evaluate(visualize=True)
cre_sco_obj.evaluate()
cre_sco_obj.tradeoff()
cre_sco_obj.feature_importance()
cre_sco_obj.compile()
result = cre_sco_obj.perf_metric_obj.result, cre_sco_obj.fair_metric_obj.result


def test_evaluate():

    assert round(result[0]['perf_metric_values']['selection_rate'][0],3) == 0.757
   
def test_artifact():
    
    assert cre_sco_obj.artifact['features']['SEX']['tradeoff']['th_x'].shape == cre_sco_obj.artifact['features']['SEX']['tradeoff']['th_y'].shape
    assert cre_sco_obj.artifact['features']['SEX']['tradeoff']['fair'].shape == cre_sco_obj.artifact['features']['SEX']['tradeoff']['perf'].shape
    assert cre_sco_obj.array_size == cre_sco_obj.artifact['perf_dynamic']['threshold'].shape[0]
    assert cre_sco_obj.array_size == len(cre_sco_obj.artifact['perf_dynamic']['perf'])
    assert cre_sco_obj.array_size == len(cre_sco_obj.artifact['perf_dynamic']['selection_rate'])
    
def test_fairness_conclusion():
    if cre_sco_obj.fair_threshold < 1:
        assert cre_sco_obj.fair_threshold == cre_sco_obj.fair_conclusion['SEX']['threshold']
    else:
        value = round((1 - cre_sco_obj.fair_conclusion['SEX']['threshold']) *100)
        assert cre_sco_obj.fair_threshold == value
    assert cre_sco_obj.fair_conclusion['SEX']['fairness_conclusion'] in ('fair','unfair')

def test_compute_fairness():
    if cre_sco_obj.fairness_metric_value_input is not None :
        assert cre_sco_obj.fairness_metric_value_input['SEX']['fpr_parity'] == cre_sco_obj.fair_metric_obj.result['SEX']['fair_metric_values']['fpr_parity'][0]
    
def test_fairness_metric_value_input_check():
    cre_sco_obj.fairness_metric_value_input = {'INCOME': {'fpr_parity': 0.2}}
    cre_sco_obj._fairness_metric_value_input_check()
    assert cre_sco_obj.fairness_metric_value_input == None
    
    cre_sco_obj.fairness_metric_value_input = {'SEX': {'other_metric': 0.2}}
    cre_sco_obj._fairness_metric_value_input_check()
    assert cre_sco_obj.fairness_metric_value_input == None
    
def test_compile():

    assert cre_sco_obj.evaluate_status == 1
    assert cre_sco_obj.evaluate_status_cali == True
    assert cre_sco_obj.evaluate_status_perf_dynamics == True
    assert cre_sco_obj.tradeoff_status == 1
    assert cre_sco_obj.feature_imp_status == 1
    assert cre_sco_obj.feature_imp_status_loo == True
    assert cre_sco_obj.feature_imp_status_corr == True
    
def test_compile_skip():
    cre_sco_obj.feature_imp_status = 0
    cre_sco_obj.tradeoff_status = 0
    cre_sco_obj.feature_imp_status_corr = False
    cre_sco_obj.compile(skip_tradeoff_flag=1, skip_feature_imp_flag=1)
    assert cre_sco_obj.feature_imp_status == -1
    assert cre_sco_obj.tradeoff_status == -1
    
def test_tradeoff():

    assert round(cre_sco_obj.tradeoff_obj.result['SEX']['max_perf_point'][0],3) == 0.407
    cre_sco_obj.model_params[0].y_prob = None
    cre_sco_obj.tradeoff()
    assert cre_sco_obj.tradeoff_status == -1
    cre_sco_obj.tradeoff_obj.result= None
    cre_sco_obj.tradeoff()
    assert cre_sco_obj.tradeoff_status == -1
    
def test_feature_importance():
    cre_sco_obj.feature_imp_status = 0
    cre_sco_obj.evaluate_status = 0
    cre_sco_obj.feature_importance()
    assert round(cre_sco_obj.feature_imp_values['SEX']['SEX'][0],3) == -0.095
    cre_sco_obj.feature_imp_status = -1
    cre_sco_obj.feature_importance()
    assert cre_sco_obj.feature_imp_values == None
    
def test_feature_importance_x_test_exception():   
    
    from veritastool.model.modelwrapper import ModelWrapper
    import numpy as np

    class xtestwrapper(ModelWrapper):
        """
        Abstract Base class to provide an interface that supports non-pythonic models.
        Serves as a template for users to define the

        """

        def __init__(self, model_obj):
            self.model_obj = model_obj
            #self.output_file = output_file
           
        """
        Parameters
        ----------
        model_file : string
                Path to the model file. e.g. "/home/model.pkl"

        output_file : string
                Path to which the prediction results will be written to in the form of a csv file. e.g. "/home/results.csv"
        """

        def fit(self, X, y):
            
            
            """
            This function is a template for user to specify a custom fit() method that trains the model and saves it to self.model_file.
            An example is as follows:
        
            train_cmd = "train_func --train {x_train} {y_train} {self.model_file}"
            import subprocess
            process = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

        """
            pass

        def predict(self, x_test, best_th = 0.43):
            error
#             test_probs = self.model_obj.predict_proba(x_test)[:, 1] 
#             test_preds = np.where(test_probs > best_th, 1, 0)
#             return test_preds

    model_object = None
    model_object = xtestwrapper(model_object)
    
    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", fair_metric_name = 'disparate_impact',fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} })
    test = cre_sco_obj.feature_importance()
    assert test == None
    
def test_feature_importance_x_train_exception():   
    
    from veritastool.model.modelwrapper import ModelWrapper
    import numpy as np

    class xtrainwrapper(ModelWrapper):
        """
        Abstract Base class to provide an interface that supports non-pythonic models.
        Serves as a template for users to define the

        """

        def __init__(self, model_obj):
            self.model_obj = model_obj
            #self.output_file = output_file
           
        """
        Parameters
        ----------
        model_file : string
                Path to the model file. e.g. "/home/model.pkl"

        output_file : string
                Path to which the prediction results will be written to in the form of a csv file. e.g. "/home/results.csv"
        """

        def fit(self, X, y):
            
            
            """
            This function is a template for user to specify a custom fit() method that trains the model and saves it to self.model_file.
            An example is as follows:
        
            train_cmd = "train_func --train {x_train} {y_train} {self.model_file}"
            import subprocess
            process = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

        """
            pass

        def predict(self, x_test, best_th = 0.43):
            #error
            test_probs = self.model_obj.predict_proba(x_test)[:, 1] 
            test_preds = np.where(test_probs > best_th, 1, 0)
            return test_preds
            
    model_object = None     
    model_object = xtrainwrapper(model_object)
    
    #Create Model Container and Use Case Object
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", fair_metric_name = 'disparate_impact',fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} })
    test = cre_sco_obj.feature_importance()
    assert test == None
        
def test_e_lift():
    result = cre_sco_obj._get_e_lift()
    assert result == None
    
def test_feature_mask():
    assert len(cre_sco_obj.model_params[0].x_test) == len(cre_sco_obj.feature_mask['SEX'])    
    
def test_base_input_check():
    cre_sco_obj.fair_metric_name = 'mi_independence'
    cre_sco_obj.fair_threshold = 43
    cre_sco_obj.fairness_metric_value_input = {'SEX': {'other_metric': 0.2}}
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj._base_input_check()
    assert toolkit_exit.type == MyError
    
def test_model_type_input():
    cre_sco_obj.model_params[0].model_type = 'svm'
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj._model_type_input()
    assert toolkit_exit.type == MyError
    
    cre_sco_obj._model_type_to_metric_lookup[cre_sco_obj.model_params[0].model_type] = ('classification', 2, 2)
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj._model_type_input()
    assert toolkit_exit.type == MyError
    
    if cre_sco_obj.model_params[0].model_type == 'uplift' and cre_sco_obj.model_params[1].model_name != "clone" :
        cre_sco_obj.model_params[1].model_name = 'duplicate'
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj._model_type_input()
    assert toolkit_exit.type == MyError
    
def test_fairness_tree():
    cre_sco_obj.fair_impact = 'normal'
    cre_sco_obj._fairness_tree()
    assert cre_sco_obj.fair_metric_name == 'equal_opportunity'
    cre_sco_obj.fair_concern = 'inclusive'
    cre_sco_obj._fairness_tree()
    assert cre_sco_obj.fair_metric_name == 'fpr_parity'
    cre_sco_obj.fair_concern = 'both'
    cre_sco_obj._fairness_tree()
    assert cre_sco_obj.fair_metric_name == 'equal_odds'
    cre_sco_obj.fair_impact = 'selective'
    cre_sco_obj.fair_concern = 'eligible'
    cre_sco_obj.fair_priority = 'benefit'
    cre_sco_obj._fairness_tree()
    assert cre_sco_obj.fair_metric_name == 'ppv_parity'
    cre_sco_obj.fair_impact = 'selective'
    cre_sco_obj.fair_concern = 'inclusive'
    cre_sco_obj.fair_priority = 'benefit'
    cre_sco_obj._fairness_tree()
    assert cre_sco_obj.fair_metric_name == 'fdr_parity'
    cre_sco_obj.fair_concern = 'both'
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj._fairness_tree()
    assert toolkit_exit.type == MyError
    cre_sco_obj.fair_impact = 'normal'
    cre_sco_obj.fair_concern = 'inclusive'
    cre_sco_obj.fair_priority = 'harm'
    cre_sco_obj._fairness_tree()
    assert cre_sco_obj.fair_metric_name == 'fpr_parity'
    
    cre_sco_obj.fair_concern = 'eligible'
    cre_sco_obj.fair_priority = 'benefit'
    cre_sco_obj.fair_impact = 'normal'
    cre_sco_obj._fairness_tree(is_pos_label_favourable = False)
    assert cre_sco_obj.fair_metric_name == 'tnr_parity'
    cre_sco_obj.fair_concern = 'inclusive'
    cre_sco_obj._fairness_tree(is_pos_label_favourable = False)
    assert cre_sco_obj.fair_metric_name == 'fnr_parity'
    cre_sco_obj.fair_concern = 'both'
    cre_sco_obj._fairness_tree(is_pos_label_favourable = False)
    assert cre_sco_obj.fair_metric_name == 'neg_equal_odds'
    cre_sco_obj.fair_impact = 'selective'
    cre_sco_obj.fair_concern = 'eligible'
    cre_sco_obj.fair_priority = 'benefit'
    cre_sco_obj._fairness_tree(is_pos_label_favourable = False)
    assert cre_sco_obj.fair_metric_name == 'npv_parity'
    cre_sco_obj.fair_impact = 'selective'
    cre_sco_obj.fair_concern = 'inclusive'
    cre_sco_obj.fair_priority = 'benefit'
    cre_sco_obj._fairness_tree(is_pos_label_favourable = False)
    assert cre_sco_obj.fair_metric_name == 'for_parity'
    cre_sco_obj.fair_concern = 'both'
    with pytest.raises(MyError) as toolkit_exit:
        cre_sco_obj._fairness_tree(is_pos_label_favourable = False)
    assert toolkit_exit.type == MyError
    cre_sco_obj.fair_impact = 'normal'
    cre_sco_obj.fair_concern = 'inclusive'
    cre_sco_obj.fair_priority = 'harm'
    cre_sco_obj._fairness_tree(is_pos_label_favourable = False)
    assert cre_sco_obj.fair_metric_name == 'fnr_parity'

    
