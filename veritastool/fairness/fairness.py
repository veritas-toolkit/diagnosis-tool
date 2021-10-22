import numpy as np
import pandas as pd
import datetime
import json
from ..util.utility import *
from ..metrics.fairness_metrics import FairnessMetrics
from ..metrics.performance_metrics import PerformanceMetrics
from ..metrics.newmetric import *
from ..metrics.tradeoff import TradeoffRate
import ipywidgets as widgets
import IPython
from ipywidgets import Layout, Button, Box, VBox, HBox, Text, GridBox
from IPython.display import display, clear_output, HTML
from IPython.core.display import HTML
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import warnings
from ..util.errors import *
from math import floor
import concurrent.futures
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.lines as mlines

class Fairness:
    """
    Base Class with attributes used across all use cases within Machine Learning model fairness evaluation.
    """
    def __init__(self, model_params):
        """
        Parameters
        ------------------
        model_params : object of type ModelContainer
                Data holder that contains all the attributes of the model to be assessed. Compulsory input for initialization.

        Instance Attributes
        -------------------
        fair_metric_obj : object, default=None
                Stores the FairnessMetrics() object and contains the result of the computations.

        perf_metric_obj : object, default=None
                Stores the PerformanceMetrics() object and contains the result of the computations.

        percent_distribution : dictionary, default=None
                Stores the percentage breakdown of the classes in y_true.

        calibration_score : float, default=None
                The brier score loss computed for calibration. Computable if y_prob is given.

        tradeoff_obj : object, default=None
                Stores the TradeoffRate() object and contains the result of the computations.

        correlation_output : dict, default=None
                Pairwise correlation of most important features (top 20 feature + protected variables).

        feature_mask : dictionary of lists, default=None
                Stores the mask array for every protected variable applied on the x_test dataset.

        fair_conclusion : dictionary, default=None
                Contains conclusion of how the primary fairness metric compares against the fairness threshold. The key will be the protected variable and the conclusion will be "fair" or "unfair".
                e.g. {"gender": {'fairness_conclusion': "fair", "threshold": 0.01}, "race":{'fairness_conclusion': "unfair", "threshold": 0.01}}

        evaluate_status : int, default=0
                Tracks the status of the completion of the evaluate() method to be checked in compile(). Either 1 for complete or -1 for error if any exceptions were raised.

        evaluate_status_cali: boolean, default=False
                Tracks the status of the completion of the calibration curve step within evaluate() method to be checked in compile().
                False = Skipped (if y_prob is not provided)
                True = Complete

        tradeoff_status : int, default=0
                Tracks the status of the completion of the tradeoff() method to be checked in compile().
                0 = Not started
                1 = Complete
                -1 = Skipped (if y_prob is not provided)

        feature_imp_status : int, default=0
                Tracks the status of the completion of the compute_feature_imp() method to be checked in compile(). 
                0 = Not started
                1 = Complete
                -1 = Skipped (if model_object not provided, wrong train_op_name/predict_op_name, x_train or x_test error)
 
        feature_imp_status_loo: boolean, default=False
                Tracks the status of the completion of the leave-one-out analysis step within feature_importance() method to be checked in compile().
                False = Skipped (if x_train or y_train or model object or fit/predict operator names are not provided)
                True = Complete

        feature_imp_status_corr: boolean, default=False
                Tracks the status of the completion of the correlation matrix computation step within feature_importance() method to be checked in compile().
                False = Skipped (if the correlation dataframe is not provided in ModelContainer)
                True = Complete

        feature_imp_values: dictionary of lists, default = None
                Contains the difference in metric values between the original and loco models for each protected variable.

                {"gender":
                         {
                          "gender": (perf_delta, fair_delta, flip, suggestion),
                          "race": (perf_delta, fair_delta, flip, suggestion)
                          },
                "race":
                         {
                          "gender": (perf_delta, fair_delta, flip, suggestion),
                          "race": (perf_delta, fair_delta, flip, suggestion)
                          }
                }

                flip = "fair to fair", "unfair to fair", "fair to unfair", "unfair to unfair"

        err : object
                VeritasError object

        """
        self.model_params = model_params
        self.fair_metric_obj = None
        self.perf_metric_obj = None
        self.percent_distribution = None
        self.calibration_score = None
        self.calibration_curve_bin = None
        self.tradeoff_obj = None
        self.correlation_output = None
        self.feature_mask = self._set_feature_mask()
        self.fair_conclusion = None
        self.evaluate_status = 0
        self.tradeoff_status = 0
        self.feature_imp_status = 0
        self.feature_imp_values = None
        self.feature_imp_status_corr = False
        self.feature_imp_status_loo = False
        self.err = VeritasError()
        
    def evaluate(self, visualize=False, output=True, n_threads=1, seed=None):
        """
        Computes the percentage count of subgroups, performance, and fairness metrics together with their confidence intervals, calibration score & fairness metric self.fair_conclusion for all protected variables.
        If visualize = True, output will be overwritten to False (will not be shown) and run fairness_widget() from Fairness.

        Parameters
        ----------
        visualize : boolean, default=False
                If visualize = True, output will be overwritten to False and run fairness_widget() from Fairness.

        output : boolean, default=True
                If output = True, _print_evaluate() from Fairness will run.

        n_threads : int, default=1
                Number of currently active threads of a job

        seed : int, default=None
                Used to initialize the random number generator.

        Returns
        ----------
        _fairness_widget() or _print_evaluate()
        """
        #check if evaluate hasn't run, only run if haven't
        if self.evaluate_status == 0:
            #to show progress bar
            eval_pbar = tqdm(total=100, desc='Evaluate performance', bar_format='{l_bar}{bar}')
            eval_pbar.update(1)
            #execute performance metrics from PerformanceMetrics class
            self._compute_performance(n_threads=n_threads, seed = seed, eval_pbar=eval_pbar)
            eval_pbar.set_description('Evaluate fairness')
            #execute fairness metrics from FairnessMetrics class
            self._compute_fairness(n_threads=n_threads, seed = seed, eval_pbar=eval_pbar)
            #to determine fairness conclusion based on inputs
            self._fairness_conclusion()
            #set status to 1 after evaluate has run
            self.evaluate_status = 1
            eval_pbar.set_description('Evaluate')
            eval_pbar.update(100 - eval_pbar.n)
            eval_pbar.close()
            print('', flush=True)
        #to trigger widget        
        if visualize == True:
            output = False
            self._fairness_widget()
        #to trigger evaluate printout
        if output == True:
            self._print_evaluate()

    def _fair_conclude(self, protected_feature_name, **kwargs):
        """
        Checks the fairness_conclusion for the selected protected feature with the primary fairness metric value against the fair_threshold
        
        Parameters
        ----------
        protected_feature_name : string
            Name of a protected feature

        Other Parameters
        ----------------
        priv_m_v : float
            Privileged metric value

        Returns
        ----------
        out : dictionary
            Fairness threshold and conclusion for the chosen protected variable
        """
        #for feature importance, when privileged metric values have been overwritten during leave-one-out analysis
        if "priv_m_v" in kwargs:
            priv_m_v = kwargs["priv_m_v"]
            value = kwargs["value"]
        #else run as per input values
        else:
            priv_m_v = self.fair_metric_obj.result.get(protected_feature_name).get("fair_metric_values").get(self.fair_metric_name)[1]
            value = self.fair_metric_obj.result[protected_feature_name]["fair_metric_values"].get(self.fair_metric_name)[0]
        
        #to handle different variations of threhold value provided e.g. float, decimals, integer
        fair_threshold = self._compute_fairness_metric_threshold(priv_m_v)

        out = {}
        #append threshold value to result
        out['threshold'] = fair_threshold

        #if metric used is ratio based, means it will either be more than 1 or less than 1. So set n = 1 to see the difference.
        if FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'ratio':
            n = 1
        #if metric used is pairty based, means it will either be more than 0 or less than 0 So set n = 0 to see the difference.
        elif FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'parity':
            n = 0

        #find absolute difference of fair values calculated after metric has been applied
        f_value = abs(value - n)
        #determine whether input values are fair or unfair depending on metrics applied
        if f_value <= fair_threshold:
            out['fairness_conclusion'] = 'fair'
        else:
            out['fairness_conclusion'] = 'unfair'
        return out

    def _fairness_conclusion(self):
        """
        Computes _fair_conclude() for all the protected features and returns results in a dictionary

        Returns
        ----------
        self.fair_conclusion : dictionary
            fair_conclusion and threshold for every protected variable
        """
        self.fair_conclusion = {}
        #to append each fair conclusion for each protected variable into a single dictionary
        for i in  self.model_params[0].p_var:
            self.fair_conclusion[i] = self._fair_conclude(i)
            

    def _compute_fairness_metric_threshold(self, priv_m_v):
        """
        Computes the fairness metric threshold based on the fair_threshold variable

        Parameters
        ----------
        priv_m_v : float
                Privileged metric value

        Returns
        ----------
        fair_threshold : float
                Fairness metric threshold
        """
        #to handle different variations of threhold value provided e.g. float, decimals, integer
        if self.fair_threshold > 1:
            self.fair_threshold = floor(self.fair_threshold)
            if FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'ratio':
                fair_threshold = 1 - (self.fair_threshold / 100)
            elif FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'parity':
                fair_threshold = (1 - (self.fair_threshold / 100)) * priv_m_v

            return fair_threshold
        else:
            return self.fair_threshold


    def _compute_performance(self, n_threads, seed, eval_pbar):
        """
        Computes the percentage count of subgroups, all the performance metrics together with their confidence intervals & the calibration curve data.

        Parameters
        -----------
        n_threads : int
                Number of currently active threads of a job

        seed : int
                Used to initialize the random number generator.

        eval_pbar : tqdm object
                Progress bar

        Returns
        ----------
        All calculations from every performance metric
        """
        #to initialize PerformanceMetrics and exceute all the perf metrics at one go
        self.perf_metric_obj = PerformanceMetrics(self)
        self.perf_metric_obj.execute_all_perf(n_threads=n_threads, seed = seed, eval_pbar=eval_pbar)
        #bring status bar to full after all perf metrics have been ran
        eval_pbar.update(1)
        #if calibration_curve function has been run, then set status to True        
        if self.perf_metric_obj.result["calibration_curve"] is None:
            self.evaluate_status_cali = False
        else:
            self.evaluate_status_cali = True 
        #if perf_dynamic function has been run, then set status to True
        if self.perf_metric_obj.result['perf_dynamic'] is None:
            self.evaluate_status_perf_dynamics = False
        else:
            self.evaluate_status_perf_dynamics = True 
        

    def _compute_fairness(self, n_threads, seed, eval_pbar):
        """
        Computes all the fairness metrics together with their confidence intervals & the self.fair_conclusion for every protected variable
        
        Parameters
        -----------
        n_threads : int
                Number of currently active threads of a job

        seed : int
                Used to initialize the random number generator.
        
        eval_pbar : tqdm object
                Progress bar

        Returns
        ----------
        All calculations from every fairness metric
        """
        #to initialize FairnessMetrics and exceute all the fair metrics at one go
        self.fair_metric_obj = FairnessMetrics(self)
        self.fair_metric_obj.execute_all_fair(n_threads=n_threads, seed = seed, eval_pbar=eval_pbar)
        #bring status bar to full after all fair metrics have been ran
        eval_pbar.update(1)
        for i in self.model_params[0].p_var:
            for j in self._use_case_metrics['fair']:
                #if user provides fair metric value input value for each protected variable
                if self.fairness_metric_value_input is not None :
                    if i in self.fairness_metric_value_input.keys(): 
                        if j in self.fairness_metric_value_input[i].keys(): 
                            self.fair_metric_obj.result[i]["fair_metric_values"][j]= (self.fairness_metric_value_input[i][j], self.fair_metric_obj.result[i]["fair_metric_values"][j][1], self.fair_metric_obj.result[i]["fair_metric_values"][j][2] )
                            msg = "{} value for {} is overwritten by user input, CI and privileged metric value may be inconsistent."
                            msg = msg.format(FairnessMetrics.map_fair_metric_to_group[j][0], i)
                            warnings.warn(msg)


    def compile(self, skip_tradeoff_flag=0, skip_feature_imp_flag=0, n_threads=1):
        """
        Runs the evaluation function together with the trade-off and feature importance sections and saves all the results to a JSON file locally.

        Parameters
        -------------
        skip_tradeoff_flag : int, default=0
                Skip running tradeoff function if it is 1.

        skip_feature_imp_flag : int, default=0
                Skip running feature importance function if it is 1.
        
        n_threads : int, default=1
                Number of currently active threads of a job

        Returns
        ----------
        Prints messages for the status of evaluate and tradeoff and generates model artifact
        """
        #check if evaluate hasn't run, only run if haven't
        if self.evaluate_status == 0:
            self.evaluate(visualize=False, output=False, n_threads=n_threads)
        #printout
        print('{:40s}{:<10}'.format('Running evaluate','done'))
        print('{:5s}{:35s}{:<10}'.format('','performance measures','done'))
        print('{:5s}{:35s}{:<10}'.format('','bias detection','done'))

        if self.evaluate_status_cali:
            print('{:5s}{:35s}{:<10}'.format('','probability calibration','done'))
        else:
            print('{:5s}{:35s}{:<10}'.format('','probability calibration','skipped'))

        if self.evaluate_status_perf_dynamics:
            print('{:5s}{:35s}{:<10}'.format('','performance dynamics','done'))
        else:
            print('{:5s}{:35s}{:<10}'.format('','performance dynamics','skipped'))
        #check if user wants to skip tradeoff, if yes tradeoff will not run, print skipped
        if self.tradeoff_status == -1:
            print('{:40s}{:<10}'.format('Running tradeoff','skipped'))
        #check if tradeoff hasn't run and user does not want to skip, only run if haven't
        elif self.tradeoff_status == 0 and skip_tradeoff_flag==0:
            try :
                self.tradeoff(output=False, n_threads=n_threads)
                #if user wants to skip tradeoff, print skipped
                if self.tradeoff_status == -1 :
                    print('{:40s}{:<10}'.format('Running tradeoff','skipped'))
                #set status to 1 after evaluate has run
                elif self.tradeoff_status == 1 :
                    print('{:40s}{:<10}'.format('Running tradeoff','done'))
            except :
                print('{:40s}{:<10}'.format('Running tradeoff','skipped'))
        #check if tradeoff hasn't run and user wants to skip, print skipped
        elif self.tradeoff_status == 0 and skip_tradeoff_flag==1:    
            self.tradeoff_status = -1 
            print('{:40s}{:<10}'.format('Running tradeoff','skipped'))
        else: 
            print('{:40s}{:<10}'.format('Running tradeoff','done'))
        #check if user wants to skip feature_importance, if yes feature_importance will not run, print skipped
        if self.feature_imp_status_corr:
            print('{:40s}{:<10}'.format('Running feature importance','done'))            
        elif self.feature_imp_status == -1:
            print('{:40s}{:<10}'.format('Running feature importance','skipped'))
        #check if feature_importance hasn't run and user does not want to skip, only run if haven't
        elif self.feature_imp_status == 0 and skip_feature_imp_flag ==0:
            try :
                self.feature_importance(output=False, n_threads=n_threads)
                if self.feature_imp_status == 1:
                    print('{:40s}{:<10}'.format('Running feature importance','done'))
                elif self.feature_imp_status_corr:
                    print('{:40s}{:<10}'.format('Running feature importance','done'))
                else:
                    print('{:40s}{:<10}'.format('Running feature importance','skipped'))
            except:
                print('{:40s}{:<10}'.format('Running feature importance','skipped'))
        #check if feature_importance hasn't run and user wants to skip, print skipped
        elif self.feature_imp_status == 0 and skip_feature_imp_flag ==1:
            self.feature_imp_status = -1
            print('{:40s}{:<10}'.format('Running feature importance','skipped'))            
        else:
            print('{:40s}{:<10}'.format('Running feature importance','done'))
        #check if feature_importance_loo has ran, if not print skipped

        if self.feature_imp_status_loo:
            print('{:5s}{:35s}{:<10}'.format('','leave-one-out analysis','done'))
        else:
            print('{:5s}{:35s}{:<10}'.format('','leave-one-out analysis','skipped'))
        #check if feature_importance_corr has ran, if not print skipped
        if self.feature_imp_status_corr:
            print('{:5s}{:35s}{:<10}'.format('','correlation analysis','done'))
        else:
            print('{:5s}{:35s}{:<10}'.format('','correlation analysis','skipped'))

        #run function to generate json model artifact file after all API functions have ran 
        self._generate_model_artifact()


    def tradeoff(self, output=True, n_threads=1):
        """
        Computes the trade-off between performance and fairness over a range  of threshold values. 
        If output = True, run the _print_tradeoff() function.

        Parameters
        -----------
        output : boolean, default=True
            If output = True, run the _print_tradeoff() function.

        n_threads : int, default=1
                Number of currently active threads of a job
        """
        #if y_prob is None, skip tradeoff
        if self.model_params[0].y_prob is None:
            self.tradeoff_status = -1
            print("Tradeoff has been skipped due to y_prob")
        #if user wants to skip tradeoff, return None            
        if self.tradeoff_status == -1:
            return
        #check if tradeoff hasn't run, only run if haven't
        elif self.tradeoff_status == 0:
            n_threads = check_multiprocessing(n_threads)
            #to show progress bar
            tdff_pbar = tqdm(total=100, desc='Tradeoff', bar_format='{l_bar}{bar}')
            tdff_pbar.update(5)
            sys.stdout.flush()
            #initialize tradeoff
            self.tradeoff_obj = TradeoffRate(self)
            
            tdff_pbar.update(10)
            #run tradeoff
            self.tradeoff_obj.compute_tradeoff(n_threads, tdff_pbar)
            tdff_pbar.update(100 - tdff_pbar.n)
            tdff_pbar.close()
            print('', flush=True)
            #if after running tradoeff, result is None, print skipped
            if self.tradeoff_obj.result == {}:
                print(self.tradeoff_obj.msg)
                self.tradeoff_status = -1
            else:
                #set status to 1 after tradeoff has ran
                self.tradeoff_status = 1
        #if tradeoff has already ran once, just print result
        if output and self.tradeoff_status == 1:
            self._print_tradeoff()

    def feature_importance(self, output=True, n_threads=1):
        """
        Trains models using the leave-one-variable-out method for each protected variable and computes the performance and fairness metrics each time to assess the impact of those variables.
        If output = True, run the _print_feature_importance() function.

        Parameters
        ------------
        output : boolean, default=True
                Flag to print out the results of evaluation in the console. This flag will be False if visualize=True.

        n_threads : int
                Number of currently active threads of a job

        Returns
        ------------
        self.feature_imp_status_loo : boolean
                Tracks the status of the completion of the leave-one-out analysis step within feature_importance() method to be checked in compile().

        self.feature_imp_status : int
                Tracks the status of the completion of the feature_importance() method to be checked in compile().

        self._compute_correlation()

        self._print_feature_importance()
        """
        #if feature_imp_status_corr hasn't run
        if self.feature_imp_status_corr == False:
            self._compute_correlation()  
        #if user wants to skip feature_importance, return None            
        if self.feature_imp_status == -1:
            self.feature_imp_values = None
            return    
            
        #check if feature_importance hasn't run, only run if haven't
        if self.feature_imp_status == 0:

            for k in self.model_params:
                x_train = k.x_train
                y_train = k.y_train
                model_object = k.model_object
                x_test = k.x_test
                train_op_name = k.train_op_name
                predict_op_name = k.predict_op_name
                # if model_object is not provided, skip feature_importance
                if model_object is None:
                    self.feature_imp_status = -1
                    print("Feature importance has been skipped due to model_object")
                    return
                else :
                    for var_name in [train_op_name, predict_op_name]:
                        #to check callable functions
                        try:
                            callable(getattr(model_object, var_name)) 
                        except:
                            self.feature_imp_status = -1
                            print("Feature importance has been skipped due to train_op_name/predict_op_name error")
                            return
            #to show progress bar
            fimp_pbar = tqdm(total=100, desc='Feature importance', bar_format='{l_bar}{bar}')
            fimp_pbar.update(1)
            self.feature_imp_values = {}
            for h in self.model_params[0].p_var:
                self.feature_imp_values[h] = {}
            fimp_pbar.update(1)

        #if evaluate_status = 0, run evaluate() first
            if self.evaluate_status == 0:
                self.evaluate(output=False)
                
            #if user wants to skip feature_importance, return None
            if self.feature_imp_status == -1:
                self.feature_imp_values = None
                return        

            fimp_pbar.update(1)

            num_p_var = len(self.model_params[0].p_var)
            n_threads = check_multiprocessing(n_threads)
            max_workers = min(n_threads, num_p_var)

            #if require to run with 1 thread, will skip deepcopy
            worker_progress = 80/num_p_var
            if max_workers >=1:
                threads = []
                with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
                    fimp_pbar.update(5)
                    #iterate through protected variables to drop one by one as part of leave-one-out
                    for i in self.model_params[0].p_var:
                        if max_workers == 1:
                            use_case_object = self
                        else:
                            use_case_object = deepcopy(self)
                        threads.append(executor.submit(Fairness._feature_imp_loo, p_variable=i, use_case_object=use_case_object, fimp_pbar=fimp_pbar, worker_progress=worker_progress ))

                    for thread in threads:
                        fimp_pbar.update(round(8/num_p_var, 2))
                        if thread.result() is None:
                            self.feature_imp_status = -1
                            return
                        else:
                            for removed_pvar, values in thread.result().items():
                                for pvar, v in values.items():
                                    self.feature_imp_values[pvar][removed_pvar] = v
            #change flag after feature_importance has finished running
            self.feature_imp_status_loo = True
            self.feature_imp_status = 1
            fimp_pbar.update(2)
            fimp_pbar.update(100.0-fimp_pbar.n)
            fimp_pbar.close()
            print('', flush=True)
        #if feature_importance has already ran once, just print result
        if output == True:
            self._print_feature_importance()


    def _feature_imp_loo(p_variable, use_case_object, fimp_pbar, worker_progress):
        """
        Maps each thread's work for feature_importance()

        Parameters
        ------------
        p_variable : str
                Name of protected variable

        use_case_object : object
                Initialised use case object

        fimp_pbar :
        
        worker_progress : 

        Returns 
        ------------
        dictionary of loo_result of each p_var
        """
        #get baseline values
        baseline_perf_values = use_case_object.perf_metric_obj.result.get("perf_metric_values").get(use_case_object.perf_metric_name)[0]
        baseline_fair_values = use_case_object.fair_metric_obj.result.get(p_variable).get("fair_metric_values").get(use_case_object.fair_metric_name)[0]
        baseline_fairness_conclusion = use_case_object.fair_conclusion.get(p_variable).get("fairness_conclusion")
        baseline_values = [baseline_perf_values, baseline_fair_values, baseline_fairness_conclusion]
        # empty y_pred_new list to be appended
        y_pred_new = []
        loo_result = {}

        # loop through model_params
        for k in range(len(use_case_object.model_params)):
           	## for uplift model type --> two model container --> need to train two models
            ## when model param len =2, then it is uplift model
            p_var = use_case_object.model_params[k].p_var
            x_train = use_case_object.model_params[k].x_train
            y_train = use_case_object.model_params[k].y_train
            model_object = use_case_object.model_params[k].model_object
            x_test = use_case_object.model_params[k].x_test
            y_pred = use_case_object.model_params[k].y_pred
            y_prob = use_case_object.model_params[k].y_prob
            pos_label = use_case_object.model_params[k].pos_label
            neg_label = use_case_object.model_params[k].neg_label
            train_op = getattr(model_object, use_case_object.model_params[k].train_op_name)
            predict_op = getattr(model_object, use_case_object.model_params[k].predict_op_name)
            #show progress bar
            fimp_pbar.update(round(worker_progress*0.9/len(use_case_object.model_params), 2))

            try:
                #check if x_train is a dataframe
                if isinstance(x_train, pd.DataFrame):
                    #drop protected variable and train model                 
                    pre_loo_model_obj = train_op(x_train.drop(columns=[p_variable]), y_train)  # train_op_name is string, need to use getattr[] to get the attribute?
                else :
                    pre_loo_model_obj = train_op(x_train, y_train, p_variable) # train_op to handle drop column i inside train_op
                    # Predict and compute performance Metrics (PerformanceMetrics.result.balanced_acc)
            except:
                #else print skipped and return None
                print("LOO analysis is skipped for [", p_variable, "] due to x_train/y_train error")
                use_case_object.feature_imp_status = -1 
                return None

            try:
                #check if x_test is a dataframe
                if isinstance(x_test, pd.DataFrame):
                    #drop protected variable and predict  
                    pre_y_pred_new = np.array(predict_op(x_test.drop(columns=[p_variable])))
                else :
                    pre_y_pred_new = predict_op(x_train, y_train, p_variable) # train_op to handle drop column i inside train_op
            except:
                #else print skipped and return None
                print("LOO analysis is skipped for [", p_variable, "] due to x_test/y_test error")
                use_case_object.feature_imp_status = -1 
                return None
            
            fimp_pbar.update(round(worker_progress*0.02, 2))
            pre_y_pred_new = predict_op(x_test.drop(columns=[p_variable]))
            #to ensure labels and datatype for predicted values are correct before running metrics
            if len(pre_y_pred_new.shape) == 1 and pre_y_pred_new.dtype.kind in ['i','O','U']:
                pre_y_pred_new, pos_label2 = check_label(pre_y_pred_new, pos_label, neg_label)
            else:
                pre_y_pred_new = pre_y_pred_new.astype(np.float64)
            y_pred_new.append(pre_y_pred_new)
        #run performance and fairness evaluation only for primary performance and fair metric
        loo_perf_value = use_case_object.perf_metric_obj.translate_metric(use_case_object.perf_metric_name, y_pred_new=y_pred_new)
        ##to find deltas (removed - baseline) for primary perf metric 
        deltas_perf = loo_perf_value - baseline_values[0]

        # to iterate through each protected variable for each protected variable that is being dropped
        for j in use_case_object.model_params[0].p_var:
            fimp_pbar.update(round(worker_progress*0.08/len(p_var), 2))
            use_case_object.fair_metric_obj.curr_p_var = j #will this work under multithreading? will not work, should changes to a copy
            ## get loo_perf_value,loo_fair_values
            loo_fair_value, loo_priv_m_v = use_case_object.fair_metric_obj.translate_metric(use_case_object.fair_metric_name, y_pred_new=y_pred_new)[:2]

            ##to find deltas (removed - baseline) for each protected variable in iteration for primary fair metric
            deltas_fair = loo_fair_value - baseline_values[1]

            ##fairness fair_conclusion
            loo_fairness_conclusion = use_case_object._fair_conclude(j, priv_m_v=loo_priv_m_v, value=loo_fair_value)
            delta_conclusion = baseline_values[2] + " to " + loo_fairness_conclusion["fairness_conclusion"]

            ##suggestion
            #if metric used is parity based, means it will either be more than 0 or less than 0. So set n = 0 to see the difference.
            if FairnessMetrics.map_fair_metric_to_group.get(use_case_object.fair_metric_name)[2] == 'parity':
                n = 0
            #if metric used is ratio based, means it will either be more than 1 or less than 1. So set n = 1 to see the difference.
            else:
                n = 1

            if (n - baseline_fair_values) * (deltas_fair) > 0:
                if PerformanceMetrics.map_perf_metric_to_group.get(use_case_object.perf_metric_name)[1] == "regression" :
                    if deltas_perf <= 0:
                        suggestion = 'exclude'
                    else:
                        suggestion = 'examine further'
                else :
                    if deltas_perf >= 0:
                        suggestion = 'exclude'
                    else:
                        suggestion = 'examine further'                    
                delta_conclusion += " (+)"
            elif (n - baseline_fair_values) * (deltas_fair) < 0:
                if PerformanceMetrics.map_perf_metric_to_group.get(use_case_object.perf_metric_name)[1] == "regression" :
                    if deltas_perf >= 0:
                        suggestion = 'include'
                    else:
                        suggestion = 'examine further'
                else:
                    if deltas_perf <= 0:
                        suggestion = 'include'
                    else:
                        suggestion = 'examine further'
                delta_conclusion += " (-)"
            else:
                if PerformanceMetrics.map_perf_metric_to_group.get(use_case_object.perf_metric_name)[1] == "regression" :
                    if deltas_perf < 0:
                        suggestion = 'exclude'
                    elif deltas_perf > 0:
                        suggestion = 'include'
                    else:
                        suggestion = 'exclude'
                else:
                    if deltas_perf > 0:
                        suggestion = 'exclude'
                    elif deltas_perf < 0:
                        suggestion = 'include'
                    else:
                        suggestion = 'exclude'                    

            loo_result[j] = [deltas_perf, deltas_fair, delta_conclusion, suggestion]

        return {p_variable: loo_result}

    def _compute_correlation(self):
        """
        Computes the top-20 correlation matrix inclusive of the protected variables
        """
        try :
            if isinstance(self.model_params[0].x_test, str):
                self.feature_imp_status_corr = False
                return
            if isinstance(self.model_params[0].feature_imp, pd.DataFrame) and isinstance(self.model_params[0].x_test, pd.DataFrame):
                #sort feature_imp dataframe by values (descending)
                sorted_dataframe = self.model_params[0].feature_imp.sort_values(by=self.model_params[0].feature_imp.columns[1], ascending=False)
                #extract n_features and pass into array
                feature_cols = np.array(sorted_dataframe.iloc[:,0])
                p_var_cols = np.array(self.model_params[0].p_var)
                feature_cols = [col for col in feature_cols if col not in p_var_cols]
                feature_cols = feature_cols[:20-len(p_var_cols)]
                #feature_columns value from x_test
                feature_columns = self.model_params[0].x_test[feature_cols]
                #p_var_columns value from protected_features_cols
                p_var_columns = self.model_params[0].x_test[p_var_cols]
                #create final columns and apply corr()
                df = pd.concat([feature_columns, p_var_columns], axis=1).corr()
                self.correlation_output = {"feature_names":df.columns.values, "corr_values":df.values}
                #return correlation_output as dataframe
                self.feature_imp_status_corr = True
            else:
                #extract n_features and pass into array
                feature_cols = np.array(self.model_params[0].x_test.columns[:20])
                p_var_cols = np.array(self.model_params[0].p_var)
                feature_cols = [col for col in feature_cols if col not in p_var_cols]
                feature_cols = feature_cols[:20-len(p_var_cols)]
                #feature_columns value from x_test
                feature_columns = self.model_params[0].x_test[feature_cols]
                #p_var_columns value from protected_features_cols
                p_var_columns = self.model_params[0].x_test[p_var_cols]
                #create final columns and apply corr()
                df = pd.concat([feature_columns, p_var_columns], axis=1).corr()
                self.correlation_output = {"feature_names":df.columns.values, "corr_values":df.values}
                self.feature_imp_status_corr = True

        except:
            self.feature_imp_status_corr = False

    def _print_evaluate(self):
        """
        Formats the results of the evaluate() method before printing to console.
        """
        if ("_rejection_inference_flag" in dir(self)):
            if True in self._rejection_inference_flag.values():
                print("Special Parameters")
                print("Rejection Inference = True")
                name = []
                for i in self.model_params[0].p_grp.keys():
                    name += [i + " - " + str(self.model_params[0].p_grp.get(i)[0])]
                    str1 = ", ".join(
                        str(e) for e in list(set(filter(lambda a: a != self.model_params[0].p_grp.get(i)[0],
                                                        self.model_params[0].protected_features_cols[i]))))
                    name += [i + " - " + str1]
                titles = ['Group', 'Base Rate', 'Number of Rejected Applicants']

                a = []
                for i in self.spl_params['base_default_rate'].keys():
                    a += self.spl_params['base_default_rate'].get(i)

                b = []
                for i in self.spl_params['num_applicants'].keys():
                    b += self.spl_params['num_applicants'].get(i)

                data = [titles] + list(zip(name, a, b))
                for i, d in enumerate(data):
                    line = '| '.join(str(x).ljust(16) for x in d)
                    print(line)
                    if i == 0:
                        print('-' * len(line))

                print("\n")

        elif hasattr(self, 'spl_params') and ('revenue' in self.spl_params or 'treatment_cost' in self.spl_params):
            print("Special Parameters")

            titles = ['Revenue', 'Treatment Cost']

            a = [self.spl_params['revenue']]

            b = [self.spl_params['treatment_cost']]

            data = [titles] + list(zip(a, b))
            for i, d in enumerate(data):
                line = '| '.join(str(x).ljust(16) for x in d)
                print(line)
                if i == 0:
                    print('-' * len(line))

            print("\n")
            
        if PerformanceMetrics.map_perf_metric_to_group.get(self.perf_metric_name)[1] != "regression":
            print("Class Distribution")
    
            if self.model_params[0].model_type != "uplift":
                print("{0:<45s}{1:>29.{decimal_pts}f}%".format("\t" + "pos_label",
                                                                self.perf_metric_obj.result.get("class_distribution").get("pos_label") * 100, decimal_pts=self.decimals))
                print("{0:<45s}{1:>29.{decimal_pts}f}%".format("\t" + "neg_label",
                                                                self.perf_metric_obj.result.get("class_distribution").get("neg_label") * 100, decimal_pts=self.decimals))
            else: 
                print("{0:<45s}{1:>29.{decimal_pts}f}%".format("\t" + "CN",
                                                                self.perf_metric_obj.result.get("class_distribution").get("CN") * 100, decimal_pts=self.decimals))
                print("{0:<45s}{1:>29.{decimal_pts}f}%".format("\t" + "TN",
                                                                self.perf_metric_obj.result.get("class_distribution").get("TN") * 100, decimal_pts=self.decimals))
                print("{0:<45s}{1:>29.{decimal_pts}f}%".format("\t" + "CR",
                                                                self.perf_metric_obj.result.get("class_distribution").get("CR") * 100, decimal_pts=self.decimals))
                print("{0:<45s}{1:>29.{decimal_pts}f}%".format("\t" + "TR",
                                                                self.perf_metric_obj.result.get("class_distribution").get("TR") * 100, decimal_pts=self.decimals))
        else:
            pass
        print("\n")

        if self.model_params[0].sample_weight is not None:
            print("Performance Metrics (Sample Weight = True)")
        else:
            print("Performance Metrics")

        def print_metric_value(metric, fair):
            v2 = " +/- "
            if fair == 0:
                if any(map(lambda x: x is None, self.perf_metric_obj.result.get("perf_metric_values")[metric])):
                    self.perf_metric_obj.result.get("perf_metric_values")[metric] = tuple(
                        'NA' if x is None else x for x in self.perf_metric_obj.result.get("perf_metric_values")[metric])
                m = "\t" + PerformanceMetrics.map_perf_metric_to_group.get(metric)[0]
                if self.perf_metric_obj.result.get("perf_metric_values").get(metric)[0] == "NA":
                    v1 = "NA"
                    v3 = "NA"
                else:   
                    v1 = "{:>0.{decimal_pts}f}".format(self.perf_metric_obj.result.get("perf_metric_values").get(metric)[0], decimal_pts=self.decimals)
                    v3 = "{:>0.{decimal_pts}f}".format(self.perf_metric_obj.result.get("perf_metric_values").get(metric)[1], decimal_pts=self.decimals)
            else:
                if any(map(lambda x: x is None, self.fair_metric_obj.result.get(i_var).get("fair_metric_values")[metric])):
                    self.fair_metric_obj.result.get(i_var).get("fair_metric_values")[metric] = tuple('NA' if x is None else x for x in self.fair_metric_obj.result.get(i_var).get("fair_metric_values")[metric])
                m = "\t" + FairnessMetrics.map_fair_metric_to_group.get(metric)[0]
                if self.fair_metric_obj.result.get(i_var).get("fair_metric_values")[metric][0] == "NA":
                    v1 = "NA"
                    v3 = "NA"
                else :
                    v1 = "{:>0.{decimal_pts}f}".format(self.fair_metric_obj.result.get(i_var).get("fair_metric_values")[metric][0], decimal_pts=self.decimals)
                    v3 = "{:>0.{decimal_pts}f}".format(self.fair_metric_obj.result.get(i_var).get("fair_metric_values")[metric][2], decimal_pts=self.decimals)

            if (v1 == "NA") & (v3 == "NA"):
                v = v1
            else:
                v = v1 + v2 + v3

            if self.perf_metric_name == metric or self.fair_metric_name == metric:
                print("\033[1m" + "{0:<45s}{1:>30s}".format(m, v) + "\033[0m")
            else:
                print("{0:<45s}{1:>30s}".format(m, v))

        for k in self._use_case_metrics["perf"]:
            print_metric_value(k, 0)

        if self.perf_metric_obj.result.get("calibration_curve") is None:
            pass
        else:
            print("\n")
            print("Probability Calibration")
            m = "\tBrier Loss Score"
            v = "{:.{decimal_pts}f}".format(self.perf_metric_obj.result.get("calibration_curve").get("score"),
                                             decimal_pts=self.decimals)

            print("{0:<45s}{1:>30s}".format(m, v))
        print("\n")

        if self.fair_metric_input == 'auto':
            print('Primary Fairness Metric Suggestion')
            print('\t{}'.format(FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[0]))
            print('based on')
            print('\tfair_priority = {}'.format(self.fair_priority))
            print('\tfair_concern = {}'.format(self.fair_concern))
            print('\tfair_impact = {}'.format(self.fair_impact))
            print('\n')
            
        for i, i_var in enumerate(self.model_params[0].p_var):
            p_len = len(str(i + 1) + ": " + i_var)
            print("-" * 35 + str(i + 1) + ": " + i_var.title() + "-" * int((45 - p_len)))

            print("Value Distribution")
            print("{:<45s}{:>29.{decimal_pts}f}%".format('\tPrivileged Group',
                                                         self.fair_metric_obj.result.get(i_var).get(
                                                             "feature_distribution").get("privileged_group") * 100,
                                                         decimal_pts=self.decimals))
            print("{:<45s}{:>29.{decimal_pts}f}%".format('\tUnprivileged Group',
                                                         self.fair_metric_obj.result.get(i_var).get(
                                                             "feature_distribution").get("unprivileged_group") * 100,
                                                         decimal_pts=self.decimals))
            print("\n")

            if self.model_params[0].sample_weight is not None:
                print("Fairness Metrics (Sample Weight = True)")
            else:
                print("Fairness Metrics")
            for h in self._use_case_metrics["fair"]:
                print_metric_value(h, 1)

            print("\n")
            print("Fairness Conclusion")
            m = "\tOutcome ({})".format(FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[0])
            v = self.fair_conclusion.get(i_var).get("fairness_conclusion").title()
            print("{0:<55s}{1:>20s}*".format(m, v))

            m = "\tFairness Threshold"

            if self.fair_threshold > 0 and self.fair_threshold < 1:
                v = str(self.fair_threshold)
            elif self.fair_threshold > 1 and self.fair_threshold < 100:
                v = str(self.fair_threshold) + "%"
            print("{0:<45s}{1:>30s}".format(m, v))
            print("\n")
        
        print('* The outcome is calculated based on your inputs and is provided for informational purposes only. Should you decide to act upon the information herein, you do so at your own risk and Veritas Toolkit will not be liable or responsible in any way. ')
        sys.stdout.flush()


    def _print_tradeoff(self):
        """
        Formats the results of the tradeoff() method before printing to console.
        """
        i = 1
        p_var = self.model_params[0].p_var
        for p_variable in p_var:
            #title
            title_str = " "+ str(i) + ". " + p_variable +" "
            if len(title_str)%2 == 1:
                title_str+=" "
            line_str = int((72-len(title_str))/2) * "-"
            print(line_str + title_str +line_str)
            
            print("Performance versus Fairness Trade-Off")
            #Single Threshold
            print("\t Single Threshold")
            print("\t\t{:35s}{:>20.{decimal_pts}f}".format("Privileged/Unprivileged Threshold",
                                                           self.tradeoff_obj.result[p_variable]["max_perf_single_th"][
                                                               0], decimal_pts=self.decimals))
            print("\t\t{:35s}{:>20.{decimal_pts}f}".format(
                str("Best " + self.tradeoff_obj.result[p_variable]["perf_metric_name"] + "*"),
                self.tradeoff_obj.result[p_variable]["max_perf_single_th"][2], decimal_pts=self.decimals))

            # Separated Thresholds
            print("\t Separated Thresholds")
            print("\t\t{:35s}{:>20.{decimal_pts}f}".format("Privileged Threshold",
                                                           self.tradeoff_obj.result[p_variable]["max_perf_point"][0],
                                                           decimal_pts=self.decimals))
            print("\t\t{:35s}{:>20.{decimal_pts}f}".format("Unprivileged Threshold",
                                                           self.tradeoff_obj.result[p_variable]["max_perf_point"][1],
                                                           decimal_pts=self.decimals))
            print("\t\t{:35s}{:>20.{decimal_pts}f}".format(
                str("Best " + self.tradeoff_obj.result[p_variable]["perf_metric_name"] + "*"),
                self.tradeoff_obj.result[p_variable]["max_perf_point"][2], decimal_pts=self.decimals))

            # Separated Thresholds under Neutral Fairness (0.01)
            print("\t Separated Thresholds under Neutral Fairness ({})".format(self.fair_neutral_tolerance))
            print("\t\t{:35s}{:>20.{decimal_pts}f}".format("Privileged Threshold", self.tradeoff_obj.result[p_variable][
                "max_perf_neutral_fair"][0], decimal_pts=self.decimals))
            print("\t\t{:35s}{:>20.{decimal_pts}f}".format("Unprivileged Threshold",
                                                           self.tradeoff_obj.result[p_variable][
                                                               "max_perf_neutral_fair"][1], decimal_pts=self.decimals))
            print("\t\t{:35s}{:>20.{decimal_pts}f}".format(
                str("Best " + self.tradeoff_obj.result[p_variable]["perf_metric_name"] + "*"),
                self.tradeoff_obj.result[p_variable]["max_perf_neutral_fair"][2], decimal_pts=self.decimals))
            print("\t\t*estimated by approximation, subject to the resolution of mesh grid")
            print("")
            i+=1
        sys.stdout.flush()

    def _print_feature_importance(self):
        """
        Formats the results of the feature_importance() method before printing to console.
        """
        for i, i_var in enumerate(self.model_params[0].p_var):
            print("\n")
            p_len = len(str(i + 1) + ": Fairness on " + i_var)
            print("-" * 50 + str(i + 1) + ": Fairness on " + i_var.title() + "-" * int((116 - 50 - p_len)))
            print()
            print("-" * 116)
            print("|{:<30}|{:<20}|{:<20}|{:<20}|{:<20}|".format("Removed Protected Variable", self.perf_metric_name,
                                                                self.fair_metric_name, "Fairness Conclusion",
                                                                "Suggestion"))
            print("-" * 116)
            for j in self.model_params[0].p_var:
                col1, col2, col3, col4 = self.feature_imp_values[i_var][j]
                print("|{:<30}|{:<20.{decimal_pts}f}|{:<20.{decimal_pts}f}|{:<20}|{:<20}|".format(j, col1, col2, col3, (col4).title(), decimal_pts=self.decimals))
                print("-" * 116)
            print()

        if self.feature_imp_status_corr == False:
            print("Correlation matrix skippped")
        else:
            return self.correlation_output
        sys.stdout.flush()

    def _generate_model_artifact(self):
        """
        Generates the JSON file to be saved locally at the end of compile()
        """
        #aggregate the results into model artifact
        print('{:40s}'.format('Generating model artifact'), end='')
        artifact = {}

        # Section 1 - fairness_init
        #write results to fairness_init
        fairness_init = {}
        fairness_init["fair_metric_name_input"] = self.fair_metric_input
        fairness_init["fair_metric_name"] = FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[0] 
        fairness_init["perf_metric_name"] = PerformanceMetrics.map_perf_metric_to_group.get(self.perf_metric_name)[0] 
        fairness_init["protected_features"] = self.model_params[0].p_var  
        if FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[1] != "regression":
            fairness_init["fair_priority"] = self.fair_priority
            fairness_init["fair_concern"] = self.fair_concern
            fairness_init["fair_impact"] = self.fair_impact
        if self.model_params[0].model_type == "uplift" or self.model_params[0].model_type == "credit": 
            fairness_init["special_params"] = self.spl_params  #num_applicants and base_default_rate for creditscoring, treatment_cost, revenue and selection_threshold for customermarketing
        fairness_init["fair_threshold_input"] = self.fair_threshold_input
        fairness_init["fair_neutral_tolerance"] = self.fair_neutral_tolerance  
        model_type = self.model_params[0].model_type  
        #add fairness_init results to artifact        
        artifact["fairness_init"] = fairness_init  
        perf_result = deepcopy(self.perf_metric_obj.result)
        perf_vals_wth_metric_names = {}
        for key in self.perf_metric_obj.result["perf_metric_values"].keys():
            if key in PerformanceMetrics.map_perf_metric_to_group.keys():
                perf_vals_wth_metric_names[PerformanceMetrics.map_perf_metric_to_group.get(key)[0]] = \
                    self.perf_metric_obj.result["perf_metric_values"][key]
        perf_result["perf_metric_values"] = perf_vals_wth_metric_names        
        artifact = {**artifact, **(perf_result)}
        artifact["correlation_matrix"] = self.correlation_output
        # above part will only be tested when Credit Scoring and Customer Marketing classes can be run
        
        p_var = self.model_params[0].p_var
        #write results to features_dict
        features_dict = {}
        for pvar in p_var:
            dic_h = {}
            dic_h["fair_threshold"] = self.fair_conclusion.get(pvar).get("threshold") 
            dic_h["privileged"] = self.model_params[0].p_grp[pvar]
            dic_t = {}
            dic_t["fairness_conclusion"] = self.fair_conclusion.get(pvar).get("fairness_conclusion")
            dic_t["tradeoff"] = None
            if self.tradeoff_status != -1:
                dic_t["tradeoff"] = self.tradeoff_obj.result.get(pvar)
            dic_t["feature_importance"] = None
            if self.feature_imp_status != -1:
                dic_t["feature_importance"] = self.feature_imp_values.get(pvar)

            fair_vals_wth_metric_names = {}
            for key in self.fair_metric_obj.result.get(pvar)['fair_metric_values'].keys():
                if key in FairnessMetrics.map_fair_metric_to_group.keys():                    
                    fair_vals_wth_metric_names[FairnessMetrics.map_fair_metric_to_group.get(key)[0]] = \
                        self.fair_metric_obj.result.get(pvar)['fair_metric_values'][key]
            fair_result = deepcopy(self.fair_metric_obj.result.get(pvar))
            fair_result['fair_metric_values'] = fair_vals_wth_metric_names
            for k, v in fair_result['fair_metric_values'].items():
                fair_result['fair_metric_values'][k] = [v[0], v[2]]
            features_dict[str(pvar)] = {**dic_h, **fair_result, **dic_t}
        #add features_dict results to artifact
        artifact["features"] = features_dict
        print('done')
        model_name = (self.model_params[0].model_name +"_").replace(" ","_")
        filename = "model_artifact_" + model_name + datetime.datetime.today().strftime('%Y%m%d_%H%M') + ".json"
        self.artifact = artifact
        artifactJson = json.dumps(artifact, cls=NpEncoder)
        jsonFile = open(filename, "w")
        jsonFile.write(artifactJson)
        jsonFile.close()
        print("Saved model artifact to " + filename)

    def _fairness_widget(self):
        """
        Runs to pop up a widget to visualize the evaluation output
        """
        try :
            if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
                display(HTML("""
                            <style>
                                .dropdown_clr {
                                    background-color: #E2F0D9;
                                }
                                .fair_green{
                                    width:auto;
                                    background-color:#E2F0D9;
                                }
                                .perf_blue {
                                    width:auto;
                                    background-color:#DEEBF7;
                                }
                            </style>
                            """))
                
                result_fairness = self.fair_metric_obj.result
                option_p_var = self.fair_metric_obj.p_var[0]
                options = []
                for i in self.fair_metric_obj.p_var[0]: 
                    options += [i + " (privileged group = " + str(self.model_params[0].p_grp.get(i))+ ")"]
                model_type = self.model_params[0].model_type.title()
                if PerformanceMetrics.map_perf_metric_to_group.get(self.perf_metric_name)[1] != "regression":
                    model_concern = self.fair_concern.title()
                    model_priority = self.fair_priority.title()
                    model_impact = self.fair_impact.title()
                else:
                    model_concern = "N/A"
                    model_priority = "N/A"
                    model_impact = "N/A"
                model_name = self.model_params[0].model_name.title()
    
                html_pink = '<div style="color:black; text-align:left; padding-left:5px; background-color:#FBE5D6; font-size:12px">{}</div>'
                html_grey_true = '<div style="color:black; text-align:center; background-color:#AEAEB2; font-size:12px">{}</div>'
                html_grey_false = '<div style="color:#8E8E93; text-align:center; background-color:#E5E5EA; font-size:12px">{}</div>'
                html_yellow_left = '<div style="color:black; text-align:left; padding-left:5px; background-color:#FFF2CC; font-size:12px">{}</div>'
                html_yellow_right = '<div style="color:black; text-align:right; padding-right:5px; background-color:#FFF2CC; font-size:12px">{}</div>'
                html_model_type = widgets.HTML(value=html_yellow_left.format('Model Type: ' + model_type),
                                            layout=Layout(display="flex", width='30%'))
                html_model_name = widgets.HTML(value=html_yellow_right.format('Model Name: ' + model_name),
                                            layout=Layout(display="flex", justify_content="flex-end", width='45%'))
                dropdown_protected_feature = widgets.Dropdown(options=options, description=r'Protected Feature:',
                                                            layout=Layout(display="flex", justify_content="flex-start",
                                                                            width='62.5%', padding='0px 0px 0px 5px'),
                                                            style=dict(description_width='initial'))
                dropdown_protected_feature.add_class("dropdown_clr")
                html_model_priority = widgets.HTML(value=html_pink.format("Priority: " + model_priority),
                                                layout=Layout(display="flex", width='12.5%'))
                html_model_impact = widgets.HTML(value=html_pink.format("Impact: " + model_impact),
                                                layout=Layout(display="flex", width='12.5%'))
                html_model_concern = widgets.HTML(value=html_pink.format('Concern: ' + model_concern),
                                                layout=Layout(display="flex", width='12.5%'))
    
                if (self.model_params[0].sample_weight is not None):
                    sw = html_grey_true
                else:
                    sw = html_grey_false
    
                if "_rejection_inference_flag" in dir(self):
                    if True in self._rejection_inference_flag.values():
                        ri = html_grey_true
                    else:
                        ri = html_grey_false
                elif hasattr(self, 'spl_params') and model_type == "Uplift":
                    if None not in self.spl_params.values():
                        ri = html_grey_true
                    else:
                        ri = html_grey_false
                else:
                    ri = html_grey_false
    
                html_sample_weight = widgets.HTML(value=sw.format('Sample Weight'),
                                                        layout=Layout(display="flex", justify_content="center", width='12.5%'))
    
                if model_type == "Credit":
                    html_rej_infer = widgets.HTML(value=ri.format('Rejection Inference'),
                                                    layout=Layout(display="flex", justify_content="center", width='12.5%'))
    
                elif model_type == "Default" or PerformanceMetrics.map_perf_metric_to_group.get(self.perf_metric_name)[1] == "regression":
                    regression = '<div style="color:#E5E5EA; text-align:center; background-color:#E5E5EA; font-size:12px">{}</div>'
                    html_rej_infer = widgets.HTML(value=regression.format('N/A'),
                                                    layout=Layout(display="flex", justify_content="center", width='12.5%'))
                elif PerformanceMetrics.map_perf_metric_to_group.get(self.perf_metric_name)[1] != "regression":
                    html_rej_infer = widgets.HTML(value=ri.format('Revenue & Cost'),
                                                    layout=Layout(display="flex", justify_content="center", width='12.5%'))
    
                html_fair_italics = '<div style="color:black; text-align:left; padding-left:5px;  font-style: italic;font-weight: bold;font-size:14px">{}</div>'
                html_fair_bold = '<div style="color:black; text-align:center;font-weight: bold;font-size:20px">{}</div>'
                html_fair_bold_red = '<div style="color:#C41E3A; text-align:center; font-weight:bold; font-size:20px">{}</div>'
                html_fair_bold_green = '<div style="color:#228B22; text-align:center; font-weight:bold; font-size:20px">{}</div>'
                html_fair_small = '<div style="color:black; text-align:left; padding-left:25px;  font-size:12px">{}</div>'
                html_fair_metric = '<div style="color:black; text-align:right;  font-weight: bold;font-size:20px">{}</div>'
                html_fair_ci = '<div style="color:black; text-align:left; padding-left:5px; font-size:15px">{}</div>'
    
                chosen_p_v = option_p_var[0]
                fair1 = widgets.HTML(value=html_fair_italics.format('Fairness'), layout=Layout(display="flex", margin='0'))
                fair2_1 = widgets.HTML(value=html_fair_small.format('Metric'),
                                    layout=Layout(display="flex", justify_content="flex-start", margin='0'))
                fair2_2 = widgets.HTML(value=html_fair_small.format('Assessment'),
                                    layout=Layout(display="flex", justify_content="flex-start", margin='0'))
    
                fair3_1 = widgets.HTML(
                    value=html_fair_bold.format(FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[0]),
                    layout=Layout(display="flex", justify_content="center", margin='0'))
                if self.fair_conclusion.get(chosen_p_v).get("fairness_conclusion") == 'fair':
                    pattern = html_fair_bold_green
                else:
                    pattern = html_fair_bold_red
                fair3_2_v = pattern.format(self.fair_conclusion.get(chosen_p_v).get("fairness_conclusion").title())
    
                fair3_2 = widgets.HTML(value=fair3_2_v,
                                    layout=Layout(display="flex", justify_content="center", margin='0'))
    
                fair4_1 = widgets.HTML(value=html_fair_small.format('Value'),
                                    layout=Layout(display="flex", justify_content="flex-start", margin='0'))
                fair4_2 = widgets.HTML(value=html_fair_small.format('Threshold'),
                                    layout=Layout(display="flex", justify_content="flex-start", margin='0'))
                v = html_fair_metric.format("{:.{decimal_pts}f}".format(self.fair_metric_obj.result.get(chosen_p_v).get('fair_metric_values').get(self.fair_metric_name)[0], decimal_pts=self.decimals))
                
                fair5_1 = widgets.HTML(value=v,layout=Layout(display="flex", width='50%', justify_content="center", margin='0'))
    
                c = html_fair_ci.format('\xB1 ' + "{:.{decimal_pts}f}".format(self.fair_metric_obj.result.get(chosen_p_v).get('fair_metric_values').get(self.fair_metric_name)[2], decimal_pts=self.decimals))
    
                fair5_1_1 = widgets.HTML(value=c,layout=Layout(display="flex", width='50%', justify_content="center", margin='0'))
                
                t = html_fair_bold.format("{:.{decimal_pts}f}".format(self.fair_conclusion.get(chosen_p_v).get("threshold"), decimal_pts=self.decimals))
                
                fair5_2 = widgets.HTML(value=t,
                                    layout=Layout(display="flex", justify_content="center", margin='0'))
    
                fair5 = HBox([fair5_1, fair5_1_1], layout=Layout(display="flex", justify_content="center"))
    
                box1f = VBox(children=[fair2_1, fair3_1, fair4_1, fair5], layout=Layout(width="66.666%"))
    
                box2f = VBox(children=[fair2_2, fair3_2, fair4_2, fair5_2], layout=Layout(width="66.666%"))
    
                box3f = HBox([box1f, box2f])
    
                box4f = VBox([fair1, box3f], layout=Layout(width="66.666%", margin='5px 5px 5px 0px'))
                box4f.add_class("fair_green")
    
                html_perf_italics = '<div style="color:black; text-align:left; padding-left:5px; font-style: italic;font-weight: bold;font-size:14px">{}</div>'
                html_perf_bold = '<div style="color:black; text-align:center;  font-weight: bold;font-size:20px">{}</div>'
                html_perf_small = '<div style="color:black; text-align:left; padding-left:25px; font-size:12px">{}</div>'
                html_perf_metric = '<div style="color:black; text-align:right; font-weight: bold;font-size:20px">{}</div>'
                html_perf_ci = '<div style="color:black; text-align:left; padding-left:5px;font-size:15px">{}</div>'
    
                perf1 = widgets.HTML(value=html_perf_italics.format('Performance'),
                                    layout=Layout(display="flex", width='33.3333%', margin='0'))
    
                perf2_1 = widgets.HTML(value=html_perf_small.format('Assessment'),
                                    layout=Layout(display="flex", justify_content="flex-start", margin='0'))
                perf3_1 = widgets.HTML(
                    value=html_perf_bold.format(PerformanceMetrics.map_perf_metric_to_group.get(self.perf_metric_name)[0]),
                    layout=Layout(display="flex", justify_content="flex-start", margin='0'))
    
                perf4_1 = widgets.HTML(value=html_perf_small.format('Value'),
                                    layout=Layout(display="flex", justify_content="flex-start", margin='0'))
                v = "{:.{decimal_pts}f}".format(self.perf_metric_obj.result.get('perf_metric_values').get(self.perf_metric_name)[0], decimal_pts=self.decimals)
                perf5_1 = widgets.HTML(value=html_perf_metric.format(v),
                                    layout=Layout(display="flex", justify_content="flex-start", width="50%", margin='0'))
                c = "{:.{decimal_pts}f}".format(self.perf_metric_obj.result.get('perf_metric_values').get(self.perf_metric_name)[1], decimal_pts=self.decimals)
                perf5_1_1 = widgets.HTML(value=html_perf_ci.format('\xB1 ' + c),
                                        layout=Layout(display="flex", justify_content="flex-start", width="50%", margin='0'))
    
                perf5 = HBox([perf5_1, perf5_1_1], layout=Layout(display="flex", justify_content="center"))
    
                box1p = VBox(children=[perf2_1, perf3_1, perf4_1, perf5])
    
                box2p = VBox([perf1, box1p], layout=Layout(width="33.333%", margin='5px 0px 5px 5px'))
    
                box2p.add_class('perf_blue')
    
                metric_box = HBox([box4f, box2p], layout=Layout(width="auto"))
    
                PATH = Path(__file__).parent.parent.joinpath('resources', 'widget')
                
                if model_type != 'Uplift' and PerformanceMetrics.map_perf_metric_to_group.get(self.perf_metric_name)[1] != "regression":
                    image1 = IPython.display.Image(filename=PATH/"perf_class_jpg.JPG", width=300, height=500)
                    A = widgets.Image(
                        value=image1.data,
                        format='jpg',
                        width=260
                    )
                    image2 = IPython.display.Image(filename=PATH/"fair_class_jpg.JPG", width=300, height=500)
                    B = widgets.Image(
                        value=image2.data,
                        format='jpg',
                        width=260
                    )
                elif model_type == "Uplift":
                    image1 = IPython.display.Image(filename=PATH/"perf_uplift_jpg.JPG", width=300, height=500)
                    A = widgets.Image(
                        value=image1.data,
                        format='jpg',
                        width=260
                    )
                    image2 = IPython.display.Image(filename=PATH/"fair_uplift_jpg.JPG", width=300, height=500)
                    B = widgets.Image(
                        value=image2.data,
                        format='jpg',
                        width=260
                    )
                else: 
                    image1 = IPython.display.Image(filename=PATH/"perf_regression_jpg.JPG", width=300, height=500)
                    A = widgets.Image(
                        value=image1.data,
                        format='jpg',
                        width=260
                    )
                    image2 = IPython.display.Image(filename=PATH/"fair_regression_jpg.JPG", width=300, height=500)
                    B = widgets.Image(
                        value=image2.data,
                        format='jpg',
                        width=260
                    )
    
                tab = widgets.Tab([A, B], layout={'width': '32%', 'margin': '15px', 'height': '350px'})
                tab.set_title(0, 'Performance Metrics')
                tab.set_title(1, 'Fairness Metrics')
                plot_output = widgets.Output(layout=Layout(display='flex', align_items='stretch', width="66.6666%"))
    
                def filtering(protected_feature):
                    global chosen_p_v
                    chosen_p_v = protected_feature
                    if self.fair_conclusion.get(chosen_p_v).get("fairness_conclusion") == 'fair':
                        fair3_2.value = html_fair_bold_green.format(self.fair_conclusion.get(chosen_p_v).get("fairness_conclusion").title())
                    else:
                        fair3_2.value = html_fair_bold_red.format(self.fair_conclusion.get(chosen_p_v).get("fairness_conclusion").title())
    
                    fair5_1.value = html_fair_metric.format("{:.{decimal_pts}f}".format(
                        self.fair_metric_obj.result.get(chosen_p_v).get('fair_metric_values').get(self.fair_metric_name)[0],
                        decimal_pts=self.decimals))
    
                    fair5_1_1.value = html_fair_ci.format('\xB1 ' + "{:.{decimal_pts}f}".format(
                        self.fair_metric_obj.result.get(chosen_p_v).get('fair_metric_values').get(self.fair_metric_name)[2],
                        decimal_pts=self.decimals))
    
                    plot_output.clear_output()
                    for metric in NewMetric.__subclasses__():
                        if metric.metric_name in result_fairness[protected_feature]['fair_metric_values'].keys():
                            del result_fairness[protected_feature]['fair_metric_values'][metric.metric_name]
                    
                    filtered_data = pd.DataFrame(result_fairness[protected_feature]['fair_metric_values'])
    
                    if model_type != 'Uplift' and PerformanceMetrics.map_perf_metric_to_group.get(self.perf_metric_name)[1] != "regression":
                        filtered_data.loc[0,'disparate_impact'] = filtered_data['disparate_impact'][0] - 1 
                    metrics = list(filtered_data.columns)
                    values = filtered_data.loc[0].values
    
                    th_min = -1*self.fair_conclusion.get(chosen_p_v).get("threshold") 
                    th_max = self.fair_conclusion.get(chosen_p_v).get("threshold") 
    
                    with plot_output:
                        fig = plt.figure(figsize=(20, 13), dpi=300)
                        clrs = ['#C41E3A' if (x == self.fair_metric_name) else '#12239E' for x in metrics]
                        ax = fig.gca()
                        idx = [i for i in range(len(values)) if values[i] == None]
                        metrics = [metrics[i] for i in range(len(metrics)) if i not in idx]
                        values = [values[i] for i in range(len(values)) if i not in idx]
                        plt.bar(metrics, values, color=clrs, align='center', width=0.5)
    
                        plt.yticks(fontsize=25)
                        label = []
                        for i in range(len(metrics)):
                            if metrics[i] == 'fpr_parity':
                                label += ["FPR Parity"]
                            elif metrics[i] == 'tnr_parity':
                                label += ["TNR Parity"]
                            elif metrics[i] == 'fnr_parity':
                                label += ["FNR Parity"]
                            elif metrics[i] == 'ppv_parity':
                                label += ["PPV Parity"]
                            elif metrics[i] == 'npv_parity':
                                label += ["NPV Parity"]
                            elif metrics[i] == 'fdr_parity':
                                label += ["FDR Parity"]
                            elif metrics[i] == 'for_parity':
                                label += ["FOR Parity"]
                            elif metrics[i] == 'mi_independence':
                                label += ["MI Independence"]
                            elif metrics[i] == 'mi_sufficiency':
                                label += ["MI Sufficiency"]
                            elif metrics[i] == 'mi_separation':
                                label += ["MI Separation"]
                            elif metrics[i] == 'disparate_impact':
                                label += ["*Disparate Impact"]
                            else:
                                label += [FairnessMetrics.map_fair_metric_to_group.get(metrics[i])[0]]
                        wrap_label = []
                        for l in label:
                            l_ = l.split(" ")
                            l_.insert(1, "\n")
                            wrap_label += [" ".join(l_)]
                        if model_type == 'Uplift' or PerformanceMetrics.map_perf_metric_to_group.get(self.perf_metric_name)[1] == "regression":
                            plt.xticks(fontsize=23, ticks=np.arange(len(label)), labels=wrap_label, rotation=0)
                        else:
                            plt.xticks(fontsize=23, ticks=np.arange(len(label)), labels=wrap_label, rotation=90)
                        ax.tick_params(axis="x", direction="in", length=16, width=2)
                        plt.ylabel("Values", fontsize=25)
                        plt.title('Fairness Metric Assessment', fontsize=35, y=1.01)
                        plt.grid(color='black', axis='y', linewidth=0.5)
    
                        plt.axhspan(th_min, th_max, color='#228B22', alpha=0.2, lw=0)
                        if max(values) > th_max:
                            ymax = max(values)*1.5
                        else:
                            ymax = th_max*1.5
    
                        if min(values) < th_min:
                            ymin = min(values)*1.5
                        else:
                            ymin = th_min*1.5
                        plt.ylim([ymin, ymax])
    
                        th = mpatches.Patch(color='#228B22', alpha=0.2,label='Threshold Range')
                        pm = mpatches.Patch(color='#C41E3A', label='Primary Metric')
                        plt.legend(handles=[pm, th],loc='upper center',  bbox_to_anchor=(0.5, -0.2),prop={"size": 25}, ncol=2, borderaxespad = 3)
                        
                        plt.box(False)
                        plt.tight_layout()
                        plt.show()
    
                def dropdown_event_handler(change):
                    new = change.new.split(" (")[0]
                    filtering(new)
    
                filtering(option_p_var[0])
                dropdown_protected_feature.observe(dropdown_event_handler, names='value')
    
                item_layout = widgets.Layout(margin='0 0 0 0')
                input_widgets1 = widgets.HBox([html_model_type, html_sample_weight, html_rej_infer, html_model_name],
                                                layout=item_layout)
                input_widgets2 = widgets.HBox([dropdown_protected_feature, html_model_priority, html_model_impact, html_model_concern],
                    layout=item_layout)
                input_widgets = VBox([input_widgets1, input_widgets2])
    
                top_display = widgets.VBox([input_widgets, metric_box])
                plot_tab = widgets.HBox([plot_output, tab])
                dashboard = widgets.VBox([top_display, plot_tab])
                display(dashboard)
                print("*The threshold and the values of ratio-based metrics are shifted down by 1.")
            else:
                print("The widget is only available on Jupyter notebook")
        except:
            pass


    def _set_feature_mask(self):
        """
        Sets the feature mask for each protected variable based on its privileged group

        Returns
        ----------
        feature_mask : dictionary of lists
                Stores the mask array for every protected variable applied on the x_test dataset.
        """
        feature_mask = {}
        for i in self.model_params[0].p_var:
            privileged_grp = self.model_params[0].p_grp.get(i)
            feature_mask[i] = self.model_params[0].protected_features_cols[i].isin(privileged_grp)  
        return feature_mask


    def _get_e_lift(self):
        """
        Helper function to get empirical lift

        Returns
        ---------
        None
        """
        return None
       
    def _get_confusion_matrix(self, curr_p_var = None,  **kwargs):
        """
        Compute confusion matrix

        Parameters
        -------------
        curr_p_var : string, default=None
                Current protected variable

        Returns
        -------
        Confusion matrix metrics based on privileged and unprivileged groups
        """
        if curr_p_var == None :
            return [None] * 4

        else :
            return [None] * 8

    def _base_input_check(self):
        """
        Checks if there are conflicting input values
        """
        try:
            if FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'information':
                if self.fair_threshold > 1:
                    self.err.push('conflict_error', var_name_a=str(self.fair_metric_name), some_string="conflict with fair_threshold", value="", function_name="_base_input_check")
                    self.err.pop()
        except TypeError:
            pass

    def _model_type_input(self):
        """
        Checks if model type input is valid
        """
        for i in self.model_params :
            #throw an error if model_type provided is not in _model_type_to_metric_lookup
            if i.model_type not in self._model_type_to_metric_lookup.keys():
                self.err.push('value_error', var_name="model_type", given=str(i.model_type),
                              expected=list(self._model_type_to_metric_lookup.keys()),
                              function_name="_model_type_input")
                #print any exceptions occured                
                self.err.pop()
  
        model_size = self._model_type_to_metric_lookup[self.model_params[0].model_type][2]
        #check if model_size provided based in model_type provided is accepted as per _model_type_to_metric_lookup
        if model_size > len(self.model_params):
                self.err.push('length_error', var_name="model_type", given=str(len(self.model_params)),
                              expected=str(model_size),
                              function_name="_model_type_input")
                #print any exceptions occured
                self.err.pop()
        #check if model_size is -1. If it is only take first set of model_params values
        elif model_size == -1:
            self.model_params = self.model_params[:1]
        else:
             self.model_params = self.model_params[:model_size]       
                
        #check if model_type of first model_container is uplift, the model_name of second model_container should be clone. Otherwise, throw an exception            
        if self.model_params[0].model_type == 'uplift':
            if self.model_params[1].model_name != "clone" :
                self.err.push('value_error', var_name="model_name", given=str(self.model_params[1].model_name),
                                  expected="clone",
                                  function_name="_model_type_input")
            #print any exceptions occured        
            self.err.pop()

    def _fairness_metric_value_input_check(self):
        """
        Checks if fairness metric value input is valid
        """
        if self.fairness_metric_value_input is not None: 
            for i in self.fairness_metric_value_input.keys() :
                #if user provided keys are not in protected variables, ignore
                if i not in self.model_params[0].p_var:
                    print("The fairness_metric_value_input is not provided properly, so it is ignored")
                    self.fairness_metric_value_input = None
                    break
                for j in self.fairness_metric_value_input[i].keys():
                    #if user provided fair metrics are not in fair metrics in use case class, ignore
                    if j not in self._use_case_metrics['fair']:
                        print("The fairness_metric_value_input is not provided properly, so it is ignored")
                        self.fairness_metric_value_input = None
                        break

    def check_fair_metric_name(self):
        """
        Checks if primary fairness metric is valid
        """
        try:
            if FairnessMetrics.map_fair_metric_to_group[self.fair_metric_name][4] == False:
                ratio_parity_metrics = []
                for i,j in FairnessMetrics.map_fair_metric_to_group.items():
                    if j[1] == self._model_type_to_metric_lookup[self.model_params[0].model_type][0]:
                        if FairnessMetrics.map_fair_metric_to_group[i][4] == True:
                            ratio_parity_metrics.append(i)
                self.err.push('value_error', var_name="fair_metric_name", given=self.fair_metric_name,  expected=ratio_parity_metrics, function_name="check_fair_metric_name")
        except:
            pass
        #print any exceptions occured
        self.err.pop()

    def check_perf_metric_name(self):
        """
        Checks if primary performance metric is valid
        """
        try:
            if PerformanceMetrics.map_perf_metric_to_group[self.perf_metric_name][4] == False:
                perf_list = []
                for i,j in PerformanceMetrics.map_perf_metric_to_group.items():
                    if j[1] == self._model_type_to_metric_lookup[self.model_params[0].model_type][0]:
                        if PerformanceMetrics.map_perf_metric_to_group[i][4] == True:
                            perf_list.append(i)
                self.err.push('value_error', var_name="perf_metric_name", given=self.perf_metric_name,  expected=perf_list, function_name="check_perf_metric_name")
        except:
            pass
        #print any exceptions occured
        self.err.pop()

    def _fairness_tree(self, is_pos_label_favourable = True):
        """
        Sets the feature mask for each protected variable based on its privileged group

        Parameters
        -----------
        is_pos_label_favourable: boolean, default=True
                Whether the pos_label is the favourable label

        Returns
        ----------
        self.fair_metric_name : string
                Fairness metric name
        """
        err_ = []

        if self.fair_concern not in ['eligible', 'inclusive', 'both']:
            err_.append(['value_error', "fair_concern", str(self.fair_concern), str(['eligible', 'inclusive', 'both'])])

        if self.fair_priority not in ['benefit', 'harm']:
            err_.append(['value_error', "fair_priority", str(self.fair_priority),str(['benefit', 'harm'])])

        if self.fair_impact not in ['significant', 'selective', 'normal']:
            err_.append(['value_error', "fair_impact", str(self.fair_impact),str(['significant', 'selective', 'normal'])])

        if err_ != []:
            for i in range(len(err_)):
                self.err.push(err_[i][0], var_name=err_[i][1], given=err_[i][2], expected=err_[i][3],
                            function_name="_fairness_tree")            
            self.err.pop()

        if is_pos_label_favourable == True:

            if self.fair_priority == "benefit":
                if self.fair_impact == "normal" :
                    if self.fair_concern == 'inclusive' :
                        self.fair_metric_name = 'fpr_parity'
                    elif self.fair_concern == 'eligible':
                        self.fair_metric_name = 'equal_opportunity'
                    elif self.fair_concern == 'both':
                        self.fair_metric_name = 'equal_odds'
                elif self.fair_impact =="significant" or self.fair_impact == "selective" :
                    if self.fair_concern == 'inclusive' :
                        self.fair_metric_name = 'fdr_parity'
                    elif self.fair_concern == 'eligible':
                        self.fair_metric_name = 'ppv_parity'
                    elif self.fair_concern == 'both':
                        self.err.push("conflict_error", var_name_a="fair_concern", some_string="not applicable", value="", function_name="_fairness_tree")
                        self.err.pop()
            elif self.fair_priority == "harm" :
                if self.fair_impact == "normal" :
                    if self.fair_concern == 'inclusive' :
                        self.fair_metric_name = 'fpr_parity'
                    elif self.fair_concern == 'eligible':
                        self.fair_metric_name = 'fnr_parity'
                    elif self.fair_concern == 'both':
                        self.fair_metric_name = 'equal_odds'
                elif self.fair_impact =="significant" or self.fair_impact == "selective" :
                    if self.fair_concern == 'inclusive' :
                        self.fair_metric_name = 'fdr_parity'
                    elif self.fair_concern == 'eligible':
                        self.fair_metric_name = 'for_parity'
                    elif self.fair_concern == 'both':
                        self.fair_metric_name = 'calibration_by_group'

        else:
            if self.fair_priority == "benefit":
                if self.fair_impact == "normal" :
                    if self.fair_concern == 'inclusive' :
                        self.fair_metric_name = 'fnr_parity'
                    elif self.fair_concern == 'eligible':
                        self.fair_metric_name = 'tnr_parity'
                    elif self.fair_concern == 'both':
                        self.fair_metric_name = 'neg_equal_odds'
                elif self.fair_impact =="significant" or self.fair_impact == "selective" :
                    if self.fair_concern == 'inclusive' :
                        self.fair_metric_name = 'for_parity'
                    elif self.fair_concern == 'eligible':
                        self.fair_metric_name = 'npv_parity'
                    elif self.fair_concern == 'both':
                        self.err.push("conflict_error", var_name_a="fairness concern", some_string="not applicable", value="", function_name="_fairness_tree")
                        self.err.pop()
            elif self.fair_priority == "harm" :
                if self.fair_impact == "normal" :
                    if self.fair_concern == 'inclusive' :
                        self.fair_metric_name = 'fnr_parity'
                    elif self.fair_concern == 'eligible':
                        self.fair_metric_name = 'fpr_parity'
                    elif self.fair_concern == 'both':
                        self.fair_metric_name = 'equal_odds'
                elif self.fair_impact =="significant" or self.fair_impact == "selective" :
                    if self.fair_concern == 'inclusive' :
                        self.fair_metric_name = 'for_parity'
                    elif self.fair_concern == 'eligible':
                        self.fair_metric_name = 'fdr_parity'
                    elif self.fair_concern == 'both':
                        self.fair_metric_name = 'calibration_by_group'
        return self.fair_metric_name


    def get_prob_calibration_results(self):
        """
        Gets the probability calibration results

        Returns
        ------------
        a dictionary with below keys and values:
            'prob_true': the ground truth values split into 10 bins from 0 to 1
            'prob_pred': the mean predicted probability in each bin
            'score': the brier loss score
        """
        if self.evaluate_status_cali == True:
            return self.perf_metric_obj.result.get("calibration_curve")
        else:
            return None

    def get_perf_metrics_results(self):
        """
        Gets the performance metrics results

        Returns
        ------------
        a dictionary with keys as the metric name and values as the metric value together with confidence interval
        """
        if self.evaluate_status == 1:
            return self.perf_metric_obj.result.get("perf_metric_values")
        else: 
            return None

    def get_fair_metrics_results(self):
        """
        Gets the fair metrics results

        Returns
        ------------
        a dictionary with keys as the metric name and values as the metric value together with confidence interval
        """
        if self.evaluate_status == 1:
            result = {}
            for p_var in self.fair_metric_obj.result.keys():
                result[p_var] = self.fair_metric_obj.result[p_var]['fair_metric_values']
            return result
        else:
            return None

    def get_tradeoff_results(self):
        """
        Gets the tradeoff results

        Returns
        ------------
        a dictionary with below keys and values:
            protected variable name as key to split result values for each protected variable
            'fair_metric_name': fairness metric name
            'perf_metric_name': performance metric name
            'fair': array of shape (n, n*) of fairness metric values
            'perf': array of shape (n, n*) of performance metric values
            'th_x': array of shape (n*, ) of thresholds on x axis
            'th_y': array of shape (n*, ) of thresholds on y axis
            'max_perf_point': maxiumn performance point on the grid
            'max_perf_single_th': maxiumn performance point on the grid with single threshold
            'max_perf_neutral_fair': maxiumn performance point on the grid with neutral fairness
            *n is defined by tradeoff_threshold_bins in config
        """
        if self.tradeoff_status == 1:
            return self.tradeoff_obj.result
        else:
            return None

    def get_loo_results(self):
        """
        Gets the leave one out analysis results

        Returns
        ------------
        a dictionary with below keys and values:
            protected variable name as key to split fairness result on each protected variable
            protected variable name as key to denote the removed protected variable
            array values denote the performance metric value, fariness metric value, fairness conclusion and suggestion
        """
        if self.feature_imp_status_loo == True:
            return self.feature_imp_values
        else:
            return None

    def get_correlation_analysis_results(self):
        """
        Gets the correlation analysis results

        Returns
        ------------
        a dictionary with below keys and values:
            'feature_names': feature names for correlation analysis
            'corr_values': correlation values according to feature names
        """
        if self.feature_imp_status_corr == True:
            return self.correlation_output
        else:
            return None

class NpEncoder(json.JSONEncoder):
    """

    """
    def default(self, obj):
        """
        Parameters
        ------------
        obj : object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

