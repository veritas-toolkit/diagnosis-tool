import pickle
from veritastool.model.model_container import ModelContainer
from veritastool.fairness.credit_scoring import CreditScoring
from veritastool.metrics.performance_metrics import PerformanceMetrics
from veritastool.metrics.fairness_metrics import FairnessMetrics
from veritastool.fairness.fairness import Fairness
from veritastool.model.modelwrapper import ModelWrapper
import numpy as np
import pandas as pd
import pytest
from veritastool.util.errors import *

# from imblearn.over_sampling import SMOTENC
# from sklearn.preprocessing import StandardScaler
# import time
# from sklearn.linear_model import LogisticRegression

# def test_LRwrapper():
    # class LRwrapper(ModelWrapper):

        # """
        # Abstract Base class to provide an interface that supports non-pythonic models.
        # Serves as a template for users to define the

        # """

        # def __init__(self, model_obj):
            # self.model_obj = model_obj
            # #self.output_file = output_file
           
        # """
        # Parameters
        # ----------
        # model_file : string
                # Path to the model file. e.g. "/home/model.pkl"

        # output_file : string
                # Path to which the prediction results will be written to in the form of a csv file. e.g. "/home/results.csv"
        # """

        # def fit(self, X, y):
        
            # #verbose and print('upsampling...')
            # categorical_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'int8']
            # smote = SMOTENC(random_state=0, categorical_features=categorical_features)
            # X, y = smote.fit_resample(X, y)

            # #verbose and print('scaling...')
            # scaling = StandardScaler()
            # X = scaling.fit_transform(X)

            # #verbose and print('fitting...')
            # #verbose and print('C:', C)
            # #model = LogisticRegression(random_state=seed, C=C, max_iter=4000)
            
            
            # self.model_obj.fit(X, y)
            # #time.sleep(30)
            
            # """
            # This function is a template for user to specify a custom fit() method that trains the model and saves it to self.model_file.
            # An example is as follows:
        
            # train_cmd = "train_func --train {x_train} {y_train} {self.model_file}"
            # import subprocess
            # process = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
            # output, error = process.communicate()

        # """
            # #pass

        # def predict(self, x_test, best_th = 0.43):
            # test_probs = self.model_obj.predict_proba(x_test)[:, 1] 
            # test_preds = np.where(test_probs > best_th, 1, 0)
            # return test_preds

            # """
            # This function is a template for user to specify a custom predict() method
            # that uses the model saved in self.model_file to make predictions on the test dataset.
        
            # Predictions can be either probabilities or labels.
        
            # An example is as follows:
        
            # pred_cmd = "pred_func --predict {self.model_file} {x_test} {self.output_file}"
            # import subprocess
            # process = subprocess.Popen(pred_cmd.split(), stdout=subprocess.PIPE)
            # output, error = process.communicate()

        # """
            # #pass
    # #Load Credit Scoring Test Data
    # #PATH = os.path.abspath(os.path.dirname(__file__))
    # file = "veritastool/resources/data/credit_score_dict.pickle"
    # input_file = open(file, "rb")
    # cs = pickle.load(input_file)

    # #Reduce into two classes
    # cs["X_train"]['MARRIAGE'] = cs["X_train"]['MARRIAGE'].replace([0, 3],1)
    # cs["X_test"]['MARRIAGE'] = cs["X_test"]['MARRIAGE'].replace([0, 3],1)
    # #Model Contariner Parameters
    # y_true = np.array(cs["y_test"])
    # y_pred = np.array(cs["y_pred"])
    # y_train = np.array(cs["y_train"])
    # p_var = ['SEX', 'MARRIAGE']
    # p_grp = {'SEX': [1], 'MARRIAGE':[1]}
    # x_train = cs["X_train"]
    # x_test = cs["X_test"]
    # #model_object = cs["model"]
    # #model_object = LogisticRegression(C=0.1)
    # model_name = "credit scoring"
    # model_type = "default"
    # y_prob = cs["y_prob"]

    # #rejection inference
    # num_applicants = {'SEX': [3500.0, 5000.0], 'MARRIAGE':[3500.0, 5000.0]}
    # base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}
    # model_object = LogisticRegression(C=0.1)
    # model_object = LRwrapper(model_object)
    # #Create Model Container and Use Case Object
    # container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    # cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", 
                               # fair_priority = "benefit", fair_impact = "significant", 
    # #                            fairness_metric_value_input = {'SEX':{'fpr_parity': 0.2} },
                              # num_applicants =num_applicants,  base_default_rate=base_default_rate)
    # # cre_sco_obj.k = 1
    # cre_sco_obj.feature_importance()
    # assert cre_sco_obj.feature_imp_values["SEX"]["SEX"][0] == -0.062067510548523164 
    # assert cre_sco_obj.feature_imp_values["SEX"]["SEX"][1] == -0.027993201728943817
    
def test_model_wrapper():
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
    #model_object = cs["model"]
    #model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]
    
    test = ModelWrapper()
    test.fit(x_train,y_train)
    test.predict(x_test)
    assert test.model_obj == None
    assert test.model_file == None
    assert test.output_file == None      
