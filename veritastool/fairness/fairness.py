import numpy as np
import pandas as pd
import datetime
import json
from ..utility import *
from .fairness_metrics import FairnessMetrics
from .performance_metrics import PerformanceMetrics
from ..custom import *
#from ..custom.newmetric_child import *
from ..fairness.tradeoff import TradeoffRate
import ipywidgets as widgets
import IPython
from ipywidgets import Layout, Button, Box, VBox, HBox, Text, GridBox
from IPython.display import display, clear_output, HTML
from IPython.core.display import HTML
import matplotlib.pyplot as plt
import os
import warnings
from ..ErrorCatcher import VeritasError
from math import floor
import concurrent.futures


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

        percent_distribution : dictionary, default = None
                Stores the percentage breakdown of the classes in y_true.

        calibration_score : float, default = None
                The brier score loss computed for calibration. Computable if y_prob is given.

        tradeoff_obj : object, default=None
                Stores the TradeoffRate() object and contains the result of the computations.

        correlation_output : Dataframe, default = None
                Pairwise correlation of most important features (top 20 feature + protected variables).

        feature_mask : dictionary of lists, default = None
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
                Tracks the status of the completion of the compute_feature_imp() method to be checked in compile(). Either 1 for complete or -1 for error if any exceptions were raised.

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

        feature_imp_status_loo: boolean, default=False
                Tracks the status of the completion of the leave-one-out analysis step within feature_importance() method to be checked in compile().
                False = Skipped (if x_train or y_train or model object or fit/predict operator names are not provided)
                True = Complete

        feature_imp_status_corr: boolean, default=False
                Tracks the status of the completion of the correlation matrix computation step within feature_importance() method to be checked in compile().
                False = Skipped (if the correlation dataframe is not provided in ModelContainer)
                True = Complete

        loo_model_obj : dictionary of objects
                Contains the models trained in feature_importance().

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
        
    def evaluate(self, visualize=False, output=True, n_threads=1):
        """
        Computes the percentage count of subgroups, performance, and fairness metrics together with their confidence intervals, calibration score & fairness metric self.fair_conclusion for all protected variables.

        Parameters
        ----------
        visualize : boolean
                If visualize = True, output will be overwritten to False and run fairness_widget() from Fairness.

        output : boolean
                If output = True, _print_evaluate() from Fairness will run.

        Returns
        ----------
        _fairness_widget() or _print_evaluate()
        """
        self._compute_performance(n_threads=n_threads)
        self._compute_fairness(n_threads=n_threads)
        self._fairness_conclusion()
        self.evaluate_status = 1
        
        if visualize == True:
            output = False
            self._fairness_widget()
        
        if output == True:
            self._print_evaluate()

    def _fair_conclude(self, protected_feature_name, **kwargs):
        """
        Checks if the primary fairness metric value against the fair_threshold for a chosen protected variable.
        Parameters
        ----------
        protected_feature_name : string
            Name of a protected feature

        Other Parameters
        ----------------
        priv_m_v : float, optional
            Privileged metric value

        Returns
        ----------
        out : dictionary
            Fairness threshold and conclusion for the chosen protected variable
        """
        if "priv_m_v" in kwargs:
            priv_m_v = kwargs["priv_m_v"]
            value = kwargs["value"]
        else:
            priv_m_v = self.fair_metric_obj.result.get(protected_feature_name).get("fair_metric_values").get(self.fair_metric_name)[1]
            value = self.fair_metric_obj.result[protected_feature_name]["fair_metric_values"].get(self.fair_metric_name)[0]
        
        fair_threshold = self._compute_fairness_metric_threshold(priv_m_v)

        out = {}
        # self.fair_conclusion['fair_metric_name'] = self.fair_metric_name
        # self.fair_conclusion['privileged_metric_value'] = priv_m_v

        # out['threshold'] = fair_threshold
        # if FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'ratio':
        #     f_value = abs(value - 1)
        #     if f_value <= fair_threshold:
        #         out['fairness_conclusion'] = 'fair'
        #     else:
        #         out['fairness_conclusion'] = 'unfair'
        # elif FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'parity':
        #     f_value = abs(value)
        #     if f_value <= fair_threshold:
        #         out['fairness_conclusion'] = 'fair'
        #     else:
        #         out['fairness_conclusion'] = 'unfair'

        out['threshold'] = fair_threshold

        if FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'ratio':
            n = 1
        elif FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'parity':
            n = 0

        f_value = abs(value - n)
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

        if self.fair_threshold > 1:
            self.fair_threshold = floor(self.fair_threshold)
            if FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'ratio':
                fair_threshold = 1 - (self.fair_threshold / 100)
            elif FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'parity':
                fair_threshold = (1 - (self.fair_threshold / 100)) * priv_m_v

            return fair_threshold
        else:
            return self.fair_threshold


    def _compute_performance(self, n_threads):
        """
        Computes the percentage count of subgroups, all the performance metrics together with their confidence intervals & the calibration curve data.

        Returns
        ----------
        All calculations from every performance metric
        """
        self.perf_metric_obj = PerformanceMetrics(self)
        self.perf_metric_obj.execute_all_perf(n_threads=n_threads)
        
        
        if self.perf_metric_obj.result["calibration_curve"] is None:
            self.evaluate_status_cali = False
        else:
            self.evaluate_status_cali = True 

        if self.perf_metric_obj.result['perf_dynamic'] is None:
            self.evaluate_status_perf_dynamics = False
        else:
            self.evaluate_status_perf_dynamics = True 
        

    def _compute_fairness(self, n_threads):
        """
        Computes all the fairness metrics together with their confidence intervals & the self.fair_conclusion for every protected variable

        Returns
        ----------
        All calculations from every fairness metric
        """
        self.fair_metric_obj = FairnessMetrics(self)
        self.fair_metric_obj.execute_all_fair(n_threads=n_threads)
        
        for i in self.model_params[0].p_var:
            for j in self._use_case_metrics['fair']:
                if self.fairness_metric_value_input is not None :
                    if i in self.fairness_metric_value_input.keys(): 
                        if j in self.fairness_metric_value_input[i].keys(): 
                            self.fair_metric_obj.result[i]["fair_metric_values"][j]= (self.fairness_metric_value_input[i][j], self.fair_metric_obj.result[i]["fair_metric_values"][j][1], self.fair_metric_obj.result[i]["fair_metric_values"][j][2] )
                            msg = "{} value for {} is overwritten by user input, CI and privileged metric value may be inconsistent."
                            msg = msg.format(FairnessMetrics.map_fair_metric_to_group[j][0], i)
                            warnings.warn(msg)


    def compile(self, skip_tradeoff_flag=0, skip_feature_imp_flag=0):
        """
        Runs the evaluation function together with the trade-off and feature importance sections and saves all the results to a JSON file locally.

        Parameters
        -------------
        skip_tradeoff_flag : int, default=0
                Skip running tradeoff function if it is 1.

        skip_feature_imp_flag : int, default=0
                Skip running feature importance function if it is 1.

        Returns
        ----------
        Prints messages for the status of evaluate and tradeoff and generates model artifact
        """
        # evaluate
        if self.evaluate_status == 0:
            print("Running evaluate", end="")
            self.evaluate(visualize=False, output=False)
            print(" \t\t\t\t done")
            print("\t performance measures \t\t\t done")
            print("\t bias detection \t\t\t done")
        else:
            print("Running evaluate \t\t\t\t done")
            print("\t performance measures \t\t\t done")
            print("\t bias detection \t\t\t done")

        if self.evaluate_status_cali:
            print("\t probability calibration \t\t done")
        else:
            print("\t probability calibration \t\t skipped")

        if self.evaluate_status_perf_dynamics:
            print("\t performance dynamics \t\t\t done")
        else:
            print("\t performance dynamics \t\t\t skipped")

        # tradeoff
        if skip_tradeoff_flag == 1:
            self.tradeoff_status = -1

        if self.tradeoff_status == -1:
            print("Running tradeoff \t\t\t\t skipped")
            
        elif self.tradeoff_status == 0:
            print("Running tradeoff", end ="")
            try :
                self.tradeoff(output=False) # add try/catch here
                if self.tradeoff_status == 1 :
                    print(" \t\t\t\t done")
                else :
                    print(" \t\t\t\t skipped")
            except :
                print(" \t\t\t\t skipped")
        else: 
            print("Running tradeoff \t\t\t\t done")

        #feature importance
        if skip_feature_imp_flag == 1:
            self.feature_imp_status = -1

        if self.feature_imp_status == -1:
            print("Running feature importance \t\t\t skipped")
        elif self.feature_imp_status == 0:
            print("Running feature importance", end ="")
            try :
                self.feature_importance(output=False) # add try/catch her
                if self.feature_imp_status == 1:
                    print(" \t\t\t done")
                else:
                    print(" \t\t\t skipped")
            except:
                 print(" \t\t\t skipped")
        else:
            print("Running feature importance \t\t\t done")

        if self.feature_imp_status_loo:
            print("\t leave-one-out analysis \t\t done")
        else:
            print("\t leave-one-out analysis \t\t skipped")
        if self.feature_imp_status_corr:
            print("\t correlation analysis \t\t\t done")
        else:
            print("\t correlation analysis \t\t\t skipped")

        # generate model artifact
        self._generate_model_artifact()


    def tradeoff(self, output=True,n_threads=0):
        """
        Computes the trade-off between performance and fairness over a range  of threshold values. If output = True, run the _print_tradeoff() function.

        Parameters
        -----------
        output : boolean
            If output = True, run the _print_tradeoff() function.

        n_threads : int, default=0
        """
        
        n_threads = check_multiprocessing(n_threads)

        #-1 => Skipped (if y_prob is not provided)
        if self.model_params[0].y_prob is None:
            self.tradeoff_status = -1
        elif self.tradeoff_status == 0:
            self.tradeoff_obj = TradeoffRate(self)
            self.tradeoff_obj.compute_tradeoff(n_threads)
            if self.tradeoff_obj.result == None:
                msg = self.tradeoff_obj.msg
                self.tradeoff_status = -1
            else:
                self.tradeoff_status = 1
                if output:
                    self._print_tradeoff()

    def feature_importance(self, output=True, n_threads=1):
        """
        Trains models using the leave-one-variable-out method for each protected variable and computes the performance and fairness metrics each time to assess the impact of those variables.

        Parameters
        ------------
        output : boolean, default=True
                Flag to print out the results of evaluation in the console. This flag will be False if visualize=True.

        Returns
        ------------
        self.feature_imp_status_loo : boolean
                Tracks the status of the completion of the leave-one-out analysis step within feature_importance() method to be checked in compile().

        self.feature_imp_status : int
                Tracks the status of the completion of the feature_importance() method to be checked in compile().

        self._compute_correlation()

        self._print_feature_importance()
        """
        if self.feature_imp_status == -1:
            self.feature_imp_values = None
            return
        if self.feature_imp_status == 0:
            self.feature_imp_values = {}
            for h in self.model_params[0].p_var:
                self.feature_imp_values[h] = {}

        # if evaluate_status = 0, call evaluate() first
            if self.evaluate_status == 0:
                # print("Running evaluate")
                self.evaluate(output=False)
        # else:
        # pass
        # print("Evaluate done")
        # iterate through protected variables to get baseline values (baseline_perf_values,baseline_fair_values,fairness_conclusion)
        ##iterate through protected variables to drop one by one as part of leave-on-out
        # fairness conclusion for baseline values
        # self._fairness_conclusion()

            num_p_var = len(self.model_params[0].p_var)
            n_threads = check_multiprocessing(n_threads)
            max_workers = min(n_threads, num_p_var)

            #if require to run with 1 thread, will skip deepcopy
            if max_workers >=1:
                #print("running parallelization with {} workers".format(str(max_workers)))
                threads = []
                with concurrent.futures.ThreadPoolExecutor(max_workers = max_workers) as executor:
                    #iterate through protected variables to drop one by one as part of leave-on-out
                    for i in self.model_params[0].p_var:
                        if max_workers == 1:
                            use_case_object = self
                        else:
                            use_case_object = deepcopy(self)
                        threads.append(executor.submit(Fairness._feature_imp_loo, p_variable = i, use_case_object = use_case_object))

                    for thread in threads:
                        if thread.result() is None:
                            print("Feature importance has been skipped due to x_test error")
                            self.feature_imp_status = -1
                            return
                        else:
                            for removed_pvar, values in thread.result().items():
                                for pvar, v in values.items():
                                    self.feature_imp_values[pvar][removed_pvar] = v

            self.feature_imp_status_loo = True
            self.feature_imp_status = 1
            self._compute_correlation()
        if output == True:
            self._print_feature_importance()


    def _feature_imp_loo(p_variable, use_case_object):
        """
        Maps each thread's work for feature_importance()
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
            # if model_param >1, then
            # check if model object is a modelwrapper? ##deepcopy will not work for model wrapper object
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

            try:
                if isinstance(x_train, pd.DataFrame):
                    pre_loo_model_obj = train_op(x_train.drop(columns=[p_variable]), y_train)  # train_op_name is string, need to use getattr[] to get the attribute?
                else :
                    pre_loo_model_obj = train_op(x_train, y_train, i) # train_op to handle drop column i inside train_op
                    # Predict and compute performance Metrics (PerformanceMetrics.result.balanced_acc)
            except:
                print("Feature importance has been skipped due to x_train error")
                use_case_object.feature_imp_status = -1 
                return None

            try:
                if isinstance(x_test, pd.DataFrame):
                    pre_y_pred_new = np.array(predict_op(x_test.drop(columns=[p_variable])))
                else :
                    pre_loo_model_obj = predict_op(x_train, y_train, i) # train_op to handle drop column i inside train_op
                    # Predict and compute performance Metrics (PerformanceMetrics.result.balanced_acc)
            except:
                print("Feature importance has been skipped due to x_test error")
                use_case_object.feature_imp_status = -1 
                return None
            #pre_y_pred_new = np.array(predict_op(x_test.drop(columns=[p_variable])))

            pre_y_pred_new = predict_op(x_test.drop(columns=[p_variable]))
            if len(pre_y_pred_new.shape) == 1 and pre_y_pred_new.dtype.kind in ['i','O','S']:
                pre_y_pred_new, pos_label2 = check_label(pre_y_pred_new, pos_label, neg_label)
            else:
                pre_y_pred_new = pre_y_pred_new.astype(np.float64)
            y_pred_new.append(pre_y_pred_new)

        # loo_values to store values for each protected variable for each protected variable that is being dropped
        # loo_values = {}
        loo_perf_value = use_case_object.perf_metric_obj.translate_metric(use_case_object.perf_metric_name, y_pred_new=y_pred_new)
        deltas_perf = loo_perf_value - baseline_values[0]

        # to iterate through each protected variable for each protected variable that is being dropped
        for j in use_case_object.model_params[0].p_var:
            use_case_object.fair_metric_obj.curr_p_var = j #will this work under multithreading? will not work, should changes to a copy

            ## get loo_perf_value,loo_fair_values
            loo_fair_value, loo_priv_m_v = use_case_object.fair_metric_obj.translate_metric(use_case_object.fair_metric_name, y_pred_new=y_pred_new)[:2]

            ##to find deltas (removed - baseline) for each protected variable in iteration
            deltas_fair = loo_fair_value - baseline_values[1]

            ##fairness fair_conclusion
            loo_fairness_conclusion = use_case_object._fair_conclude(j, priv_m_v=loo_priv_m_v, value=loo_fair_value)
            delta_conclusion = baseline_values[2] + " to " + loo_fairness_conclusion["fairness_conclusion"]

            ##suggestion
            if FairnessMetrics.map_fair_metric_to_group.get(use_case_object.fair_metric_name)[2] == 'parity':
                n = 0
            else:
                n = 1

            if (n - baseline_fair_values) * (deltas_fair) > 0:
                if deltas_perf >= 0:
                    suggestion = 'exclude'
                else:
                    suggestion = 'examine further'
                delta_conclusion += " (+)"
            elif (n - baseline_fair_values) * (deltas_fair) < 0:
                if deltas_perf <= 0:
                    suggestion = 'include'
                else:
                    suggestion = 'examine further'
                delta_conclusion += " (-)"
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

        Returns
        --------
        self.correlation_output : dictionary
            Saves correlation metrics to a dictionary

        self.feature_imp_status_corr : boolean
            if successfully computed, self.feature_imp_status_corr = True
        """
        #check if x_test is a DataFrame
        
        if isinstance(self.model_params[0].feature_imp, pd.DataFrame) and isinstance(self.model_params[0].x_test, pd.DataFrame):
            #if its a dataframe can we retrive the keys
            #try:
                #self.model_params[0].feature_imp.keys()
            #except:
                #print("Unable to retrieve dataframe keys")
            #sort feature_imp dataframe by values (descending)
            sorted_dataframe = self.model_params[0].feature_imp.sort_values(by=self.model_params[0].feature_imp.columns[1], ascending=False)
            #extract n_features and pass into array
            feature_cols = np.array(sorted_dataframe.iloc[:,0])
            #extract protected variables and pass into array
            p_var_cols = np.array(self.model_params[0].p_var)
            #create final array
            final_cols = np.append(feature_cols,[x for x in p_var_cols if x not in feature_cols])
            #filter final_cols on x_test and apply corr()
            df = self.model_params[0].x_test[final_cols].corr()
            self.correlation_output = {"feature_names":df.columns.values, "corr_values":df.values}
            #return correlation_output as dataframe
            self.feature_imp_status_corr = True
            
        else:
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

        elif 'revenue' in self.spl_params or 'treatment_cost' in self.spl_params:
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
        print("Class Distribution")

        for i in self.perf_metric_obj.result.get("class_distribution"):
            print("{0:<35s}{1:>29.{decimal_pts}f}%".format("\t" + i,
                                                           self.perf_metric_obj.result.get("class_distribution").get(
                                                               i) * 100, decimal_pts=self.decimals))

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
                    # print(self.perf_metric_obj.result.get("perf_metric_values").get(metric)[0])
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
                print("\033[1m" + "{0:<35s}{1:>30s}".format(m, v) + "\033[0m")
            else:
                print("{0:<35s}{1:>30s}".format(m, v))

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

            print("{0:<35s}{1:>30s}".format(m, v))

        for i, i_var in enumerate(self.model_params[0].p_var):
            print("\n")
            p_len = len(str(i + 1) + ": " + i_var)
            print("-" * 35 + str(i + 1) + ": " + i_var.title() + "-" * int((35 - p_len)))

            print("Value Distribution")
            print("{:<35s}{:>29.{decimal_pts}f}%".format('\tPrivileged Group',
                                                         self.fair_metric_obj.result.get(i_var).get(
                                                             "feature_distribution").get("privileged_group") * 100,
                                                         decimal_pts=self.decimals))
            print("{:<35s}{:>29.{decimal_pts}f}%".format('\tUnprivileged Group',
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
            print("Conclusion")
            m = "\tFairness Conclusion"
            v = self.fair_conclusion.get(i_var).get("fairness_conclusion").title()
            print("{0:<35s}{1:>30s}".format(m, v))

            m = "\tFairness Threshold"

            if self.fair_threshold > 0 and self.fair_threshold < 1:
                v = str(self.fair_threshold)
            elif self.fair_threshold > 1 and self.fair_threshold < 100:
                v = str(self.fair_threshold) + "% (p-% rule)"
            print("{0:<35s}{1:>30s}".format(m, v))


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
            print("\t\t*the best metric value is sub-optimal, subject to the resolution of mesh grid")
            print("")
            i+=1

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

    def _generate_model_artifact(self):
        """
        Generates the JSON file to be saved locally at the end of compile()
        """
        # aggregate the results into model artifact
        print("Generating model artifact", end="")
        artifact = {}

        # Section 1 - fairness_init

        # below part will only be tested when Credit Scoring and Customer Marketing classes can be run
        fairness_init = {}
        fairness_init["fair_metric_name"] = self.fair_metric_name  # from usecase.fair_metric_name
        fairness_init["perf_metric_name"] = self.perf_metric_name  # from usecase.perf_metric_name
        fairness_init["protected_features"] = self.model_params[0].p_var  # from model_params[0].p_var
        fairness_init["fair_neutral_tolerance"] = self.fair_neutral_tolerance  # usecase.fair_neutral_tolerance
        model_type = self.model_params[0].model_type  # from model_params[0].model_type
        fairness_init["special_params"] = self.spl_params  # num_applicants and base_default_rate for CS, treatment_cost, revenue and selection_threshold for CM
        artifact["fairness_init"] = fairness_init  # from model_param and other places

        artifact = {**artifact, **(self.perf_metric_obj.result)}
        artifact["correlation_matrix"] = self.correlation_output
        # above part will only be tested when Credit Scoring and Customer Marketing classes can be run

        p_var = self.model_params[0].p_var
        features_dict = {}
        for pvar in p_var:
            dic_h = {}
            dic_h["fair_threshold"] = self.fair_threshold
            dic_h["privileged"] = self.model_params[0].p_grp[pvar]
            dic_t = {}
            dic_t["fairness_conclusion"] = self.fair_conclusion.get(pvar).get("fairness_conclusion")
            dic_t["tradeoff"] = None
            if self.tradeoff_status != -1:
                dic_t["tradeoff"] = self.tradeoff_obj.result.get(pvar)
            dic_t["feature_importance"] = None
            if self.feature_imp_status != -1:
                dic_t["feature_importance"] = self.feature_imp_values.get(pvar)
            #fair_metric_obj
            fair_result = deepcopy(self.fair_metric_obj.result.get(pvar))
            for k, v in fair_result['fair_metric_values'].items():
                fair_result['fair_metric_values'][k] = [v[0], v[2]]
            features_dict[str(pvar)] = {**dic_h, **fair_result, **dic_t}
        artifact["features"] = features_dict
        print(" \t\t\t done")
        model_name = self.model_params[0].model_name
        model_name = model_name.replace(" ","_")
        # filename format model_artifact_{model_name1_model_name2}_yyyymmdd_hhmm.json
        filename = "model_artifact_" + model_name + datetime.datetime.today().strftime('_%Y%m%d_%H%M') + ".json"
        self.artifact = artifact
        artifactJson = json.dumps(artifact, cls=NpEncoder)
        jsonFile = open(filename, "w")
        jsonFile.write(artifactJson)
        jsonFile.close()
        print("Saved model artifact to " + filename)

    def _fairness_widget(self):
        """
        Runs to pop up a widget to visualize the evaluation output

        Returns
        --------
        Widget that displays the plot of fairness metrics and the performance and fairness metric values and threshold of selected metric
        """
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
        options = self.fair_metric_obj.p_var[0]
        model_type = self.model_params[0].model_type.title()
        model_concern = self.fair_concern.title()
        model_priority = self.fair_priority.title()
        model_impact = self.fair_impact.title()
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

        if (self.model_params[0].sample_weight != None):
            sw = html_grey_true
        else:
            sw = html_grey_false

        if "_rejection_inference_flag" in dir(self):
            if True in self._rejection_inference_flag.values():
                ri = html_grey_true
            else:
                ri = html_grey_false
        else:
            ri = html_grey_false

        html_sample_weight = widgets.HTML(value=sw.format('Sample Weight'),
                                              layout=Layout(display="flex", justify_content="center", width='12.5%'))
        html_rej_infer = widgets.HTML(value=ri.format('Rejection Inference'),
                                          layout=Layout(display="flex", justify_content="center", width='12.5%'))


        html_fair_italics = '<div style="color:black; text-align:left; padding-left:5px;  font-style: italic;font-weight: bold;font-size:14px">{}</div>'
        html_fair_bold = '<div style="color:black; text-align:center;font-weight: bold;font-size:20px">{}</div>'
        html_fair_bold_red = '<div style="color:#C41E3A; text-align:center; font-weight:bold; font-size:20px">{}</div>'
        html_fair_bold_green = '<div style="color:#228B22; text-align:center; font-weight:bold; font-size:20px">{}</div>'
        html_fair_small = '<div style="color:black; text-align:left; padding-left:25px;  font-size:12px">{}</div>'
        html_fair_metric = '<div style="color:black; text-align:right;  font-weight: bold;font-size:20px">{}</div>'
        html_fair_ci = '<div style="color:black; text-align:left; padding-left:5px; font-size:15px">{}</div>'

        chosen_p_v = options[0]
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


        PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

        if model_type != 'Uplift':
            image1 = IPython.display.Image(filename=PATH + "\\resources\\widget\\perf_class_jpg.jpg", width=300, height=500)
            A = widgets.Image(
                value=image1.data,
                format='jpg',
                width=260
            )

            image2 = IPython.display.Image(filename=PATH + "\\resources\\widget\\fair_class_jpg.jpg", width=300, height=500)
            B = widgets.Image(
                value=image2.data,
                format='jpg',
                width=260
            )
        else:
            image1 = IPython.display.Image(filename=PATH + "\\resources\\widget\\perf_uplift_jpg.jpg", width=300, height=500)
            A = widgets.Image(
                value=image1.data,
                format='jpg',
                width=260
            )

            image2 = IPython.display.Image(filename=PATH + "\\resources\\widget\\fair_uplift_jpg.jpg", width=300, height=500)
            B = widgets.Image(
                value=image2.data,
                format='jpg',
                width=260
            )

        tab = widgets.Tab([A, B], layout={'width': '32%', 'margin': '15px', 'height': '320px'})
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

            filtered_data_ = pd.DataFrame(result_fairness[protected_feature]['fair_metric_values'])
            filtered_data = filtered_data_[filtered_data_.columns[~filtered_data_.columns.isin(['disparate_impact'])]]
            metrics = list(filtered_data.columns)
            values = filtered_data.loc[0].values

            with plot_output:
                fig = plt.figure(figsize=(20, 10), dpi=300)
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
                    else:
                        label += [FairnessMetrics.map_fair_metric_to_group.get(metrics[i])[0]]
                wrap_label = []
                for l in label:
                    l_ = l.split(" ")
                    l_.insert(1, "\n")
                    wrap_label += [" ".join(l_)]
                if model_type == 'Default':
                    plt.xticks(fontsize=23, ticks=np.arange(len(label)), labels=wrap_label, rotation=90)
                else:
                    plt.xticks(fontsize=23, ticks=np.arange(len(label)), labels=wrap_label, rotation=0)
                ax.tick_params(axis="x", direction="in", length=16, width=2)
                plt.ylabel("Values", fontsize=25)
                plt.title('Fairness Metric Assessment', fontsize=35)
                plt.grid(color='black', axis='y', linewidth=0.5)
                # plt.gcf().text(10, 16, "threshold marked as"+'\u25b2', fontsize=20)
                # plt.title("Threshold marked as "+'\u25b2', loc='right', fontsize=25)
                if self.fair_metric_name != "disparate_impact":
                    i = metrics.index(self.fair_metric_name)
                    plt.plot(metrics[i], self.fair_conclusion.get(chosen_p_v).get("threshold"), c='#FFA500', marker="^",
                             markersize=25, label='Threshold')
                    plt.legend(bbox_to_anchor=(1, 1), loc='lower right', prop={"size": 25})

                plt.box(False)
                plt.tight_layout()
                plt.show()

        def dropdown_event_handler(change):
            filtering(change.new)

        filtering(options[0])
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
        #check if mutual info is set as fairness_metric_name

        # errMsg = "data value error"
        # errMsgFormat = "\n   {:.0f}%-rule does not apply to {}"
        try:
            if FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2] == 'information':
                if self.fair_threshold > 1:
                # errMsg += errMsgFormat.format(self.fair_threshold, str(self.fair_metric_name))
                # raise ValueError(errMsg)
                    self.err.push('conflict_error', var_name_a=str(self.fair_metric_name), some_string="conflict with fair_threshold", value="", function_name="_base_input_check")
                    self.err.pop()
        except TypeError:
            pass


    def _model_type_input(self):

        """
        Checks if model type input is valid
        """
        # errMsg ="data value error"
        # errMsgFormat = "\n    {}: given {}, expected {}"

        for i in self.model_params :
            if i.model_type not in self._model_type_to_metric_lookup.keys():

                # errMsg += errMsgFormat.format("model_type", str(i.model_type), list(self._model_type_to_metric_lookup.keys()))
                # raise ValueError(errMsg)
                self.err.push('value_error', var_name="model_type", given=str(i.model_type),
                              expected=list(self._model_type_to_metric_lookup.keys()),
                              function_name="_model_type_input")
                
                self.err.pop()
                # errMsg += errMsgFormat.format("model_type", str(i.model_type), list(self._model_type_to_metric_lookup.keys()))
                # raise ValueError(errMsg)    

        model_size = self._model_type_to_metric_lookup[self.model_params[0].model_type][2]
        if model_size > len(self.model_params):
                # errMsg += errMsgFormat.format("length of model params", str(len(self.model_params)), len(model_size))
                self.err.push('value_error', var_name="model_type", given=str(len(self.model_params)),
                              expected=str(model_size),
                              function_name="_model_type_input")
                self.err.pop()
        else:
             self.model_params = self.model_params[:model_size]       

        
                
                
        if self.model_params[0].model_type == 'uplift' and self.model_params[1].model_name != "clone" :
            # errMsg += errMsgFormat.format("model_name", str(self.model_params[1].model_name), "clone")
            # raise ValueError(errMsg)    
            self.err.push('value_error', var_name="model_name", given=str(self.model_params[1].model_name),
                              expected="clone",
                              function_name="_model_type_input")
        
            self.err.pop()


            
    def _fairness_metric_value_input_check(self):
        """
        Checks if fairness metric value input is valid
        """
        # errMsg ="data value error"
        # errMsgFormat = "\n    {}: given {}, expected {}"
        if self.fairness_metric_value_input is not None: 
            for i in self.fairness_metric_value_input.keys() :
                if i not in self.model_params[0].p_var:
                    # errMsg += errMsgFormat.format("fairness_metric_value_input protected variable", i, self.model_params[0].p_var)
                    # raise ValueError(errMsg)
                    self.err.push('value_error', var_name="fairness_metric_value_input protected variable", given=i, expected=self.model_params[0].p_var, function_name="_fairness_metric_value_input_check")
                for j in self.fairness_metric_value_input[i].keys():
                    if j not in self._use_case_metrics['fair']:
                        # errMsg += errMsgFormat.format("fairness_metric_value_input fairness metric name", str(j), list(self._use_case_metrics['fair']))
                        # raise ValueError(errMsg)
                        self.err.push('value_error', var_name="fairness_metric_value_input fairness metric name", given=str(j), expected=list(self._use_case_metrics['fair']), function_name="_fairness_metric_value_input_check")
            self.err.pop()

    def _fairness_tree(self, is_pos_label_favourable = True):
        """
        Sets the feature mask for each protected variable based on its privileged group

        Parameters
        -----------
        is_pos_label_favourable: boolean, default=True

        Returns
        ----------
        self.fair_metric_name : string
                Fairness metric name
        """
        # msg = "The fairness concern of 'both' is not applicable to these criteria."
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
                    # self.fair_concern_check()
                    elif self.fair_concern == 'both':
                        self.err.push("conflict_error", var_name_a="fairness concern", some_string="not applicable", value="", function_name="_fairness_tree")
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
                    # self.fair_concern_check()
                    elif self.fair_concern == 'both':
                        self.err.push("conflict_error", var_name="fairness concern", some_string="not applicable", value="", function_name="_fairness_tree")
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
 
