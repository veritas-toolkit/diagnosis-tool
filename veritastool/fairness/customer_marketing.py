import numpy as np
from sklearn.metrics import confusion_matrix
from .fairness import Fairness
from ..util.utility import check_datatype, check_value
from ..metrics.performance_metrics import PerformanceMetrics
from ..metrics.fairness_metrics import FairnessMetrics
from ..config.constants import Constants
from ..util.errors import *

class CustomerMarketing(Fairness):
    """
    A class to evaluate and analyse fairness in customer marketing related applications.

    Class Attributes
    ------------------
    _model_type_to_metric_lookup: dictionary
                Used to associate the model type (key) with the metric type, expected size of positive and negative labels (value) & length of model_params respectively.
                e.g. {“rejection”: (“classification”, 2, 1), “uplift”: (“uplift”, 4, 2), “a_new_type”: (“regression”, -1, 1)}

    """
    _model_type_to_metric_lookup = {"uplift":("uplift", 4, 2),
                                   "rejection": ("classification", 2, 1),
                                   "propensity":("classification", 2, 1)}

    def __init__(self, model_params, fair_threshold, perf_metric_name =  "balanced_acc", fair_metric_name = "auto",  fair_concern = "eligible", fair_priority = "benefit", fair_impact = "normal", treatment_cost = None, revenue = None, fairness_metric_value_input = {}, proportion_of_interpolation_fitting = 1.0):
        """
        Parameters
        ----------
        model_params: list containing ModelContainer object(s)
                Data holder that contains all the attributes of the model to be assessed. Compulsory input for initialization.
                If a single object is provided, it will be taken as either a "rejection" or "propensity" model according to the model_type flag.
                If 2 objects are provided, while the model_type flag is "uplift", the first one corresponds to rejection model while the second one corresponds to propensity model.
                **x_train[0] = x_train[1] and x_test[0]=x_test[1] must be the same when len(model_param) > 1

        fair_threshold: int or float
                Value between 0 and 100. If a float between 0 and 1 (not inclusive) is provided, it is used to benchmark against the primary fairness metric value to determine the fairness_conclusion.
                If an integer between 1 and 100 is provided, it is converted to a percentage and the p % rule is used to calculate the fairness threshold value.

        Instance Attributes
        ------------------
        perf_metric_name: string, default = "balanced_acc"
                Name of the primary performance metric to be used for computations in the evaluate() and/or compile() functions.

        fair_metric_name : string, default = "auto"
                Name of the primary fairness metric to be used for computations in the evaluate() and/or compile() functions

        fair_concern: string, default = "eligible"
               Used to specify a single fairness concern applied to all protected variables. Could be "eligible" or "inclusive" or "both".

        fair_priority: string, default = "benefit"
                Used to pick the fairness metric according to the Fairness Tree methodology. Could be "benefit" or "harm"

        fair_impact: string, default = "normal"
                Used to pick the fairness metric according to the Fairness Tree methodology. Could be "normal" or "significant" or "selective"

        treatment_cost: int or float, default=None
                Cost of the marketing treatment per customer

        revenue: int or float, default=None
                Revenue gained per customer

        fairness_metric_value_input : dictionary
                Contains the p_var and respective fairness_metric and value 
                e.g. {"gender": {"fnr_parity": 0.2}}

        proportion_of_interpolation_fitting : float, default=1.0
                Proportion of interpolation fitting

        _use_case_metrics: dictionary of lists, default=None
                Contains all the performance & fairness metrics for each use case.
                e.g. {"fair ": ["fnr_parity", ...], "perf": ["balanced_acc, ..."]}
                Dynamically assigned during initialisation by using the _metric_group_map in Fairness/Performance Metrics class and the _model_type_to_metric above.

        _input_validation_lookup: dictionary
                Contains the attribute and its correct data type for every argument passed by user. Used to perform the Utility checks.
                e.g. _input_validation_lookup = {
                "fair_threshold": [(float,), (int(config.get('threshold','fair_threshold_low')), int(config.get('threshold','fair_threshold_high')))],
                "fair_neutral_tolerance": [(float,) ,(int(config.get('threshold','fair_neutral_tolerance_low')), float(config.get('threshold','fair_neutral_tolerance_high')))],
                "sample_weight": [(int,), (0, np.inf)],
                "perf_metric_name": [(str,), _use_case_metrics["perf"]],
                "fair_metric_name": [(str,), _use_case_metrics["fair"]],
                "concern": [(str,), ["eligible", "inclusion", "both"]]
                }

        k : int
                Integer from Constants class to calculate confidence interval

        array_size : int
                Integer from Constants class to fix array size

        decimals : int
                Integer from Constants class to fix number of decimals to round off

        err : object
                VeritasError object

        selection_threshold : float
                Selection threshold from Constants class

        e_lift : float
                Empirical lift

        pred_outcome: dictionary
                Contains the probabilities of the treatment and control groups for both rejection and acquiring
        """
        super().__init__(model_params)

        self.fair_metric_name = fair_metric_name
        self.fair_metric_input = fair_metric_name
        self.perf_metric_name = perf_metric_name
        self.fair_threshold = fair_threshold
        self.fair_threshold_input = fair_threshold
        self.fair_neutral_tolerance = Constants().fair_neutral_tolerance 
        self.fair_concern = fair_concern
        self.fair_priority = fair_priority
        self.fair_impact = fair_impact
        self.fairness_metric_value_input = fairness_metric_value_input
        self.proportion_of_interpolation_fitting = proportion_of_interpolation_fitting

        self._model_type_input()

        self._select_fairness_metric_name()
        self.check_perf_metric_name()
        self.check_fair_metric_name()

        self._use_case_metrics = {}

        use_case_fair_metrics = []
        for i,j in FairnessMetrics.map_fair_metric_to_group.items():
            if j[1] == self._model_type_to_metric_lookup[self.model_params[0].model_type][0]:
                use_case_fair_metrics.append(i)
        self._use_case_metrics["fair"] = use_case_fair_metrics

        use_case_perf_metrics = []
        for i, j in PerformanceMetrics.map_perf_metric_to_group.items():
            if j[1] == self._model_type_to_metric_lookup[self.model_params[0].model_type][0]:
                use_case_perf_metrics.append(i)
        self._use_case_metrics["perf"] = use_case_perf_metrics
         # Append new performance metrics from custom class to the above list (if it is relevant)
        
        self._input_validation_lookup = {
            "fair_threshold": [(float, int), (Constants().fair_threshold_low), Constants().fair_threshold_high],
            "fair_neutral_tolerance": [(float,),(Constants().fair_neutral_threshold_low), Constants().fair_neutral_threshold_high],
            "proportion_of_interpolation_fitting": [(float,), (Constants().proportion_of_interpolation_fitting_low), Constants().proportion_of_interpolation_fitting_high],
            "fair_concern": [(str,), ["eligible", "inclusive", "both"]],
            "fair_priority": [(str,), ["benefit", "harm"]],
            "fair_impact": [(str,), ["normal", "significant", "selective"]],
            "perf_metric_name": [(str,), self._use_case_metrics["perf"]],
            "fair_metric_name": [(str,), self._use_case_metrics["fair"]],
            "model_params":[(list,), None],
            "fairness_metric_value_input":[(dict,), None]}

        self.k = Constants().k
        self.array_size = Constants().perf_dynamics_array_size
        self.decimals = Constants().decimals

        if self.model_params[0].model_type == "uplift":
            self.spl_params = {'revenue': revenue, 'treatment_cost': treatment_cost}
        self.selection_threshold = Constants().selection_threshold

        self._check_input()

        self.e_lift = self._get_e_lift()
        self.pred_outcome = self._compute_pred_outcome()

    def _check_input(self):
        """
        Wrapper function to perform all checks using dictionaries of datatypes & dictionary of values.
        This function does not return any value. Instead, it raises an error when any of the checks from the Utility class fail.
        """
        #import error class
        err = VeritasError()
        #check datatype of input variables to ensure they are of the correct datatype
        check_datatype(self)

        #check datatype of input variables to ensure they are reasonable
        check_value(self)

        #check for length of model_params
        mp_g = len(self.model_params)
        mp_e = self._model_type_to_metric_lookup[self.model_params[0].model_type][2]
        if mp_g != mp_e:
            err.push('length_error', var_name="model_params", given=str(mp_g), expected= str(mp_e), function_name="_check_input")

        #check for conflicting input values
        self._base_input_check()

        #check if input variables will the correct fair_metric_name based on fairness tree
        self._fairness_metric_value_input_check()

        #check for y_prob not None if model is uplift, else check for y_pred not None
        if self.model_params[0].model_type  == "uplift":
            for i in range(len(self.model_params)):
                if self.model_params[i].y_prob is None:
                    err.push('type_error', var_name="y_prob", given= "type None", expected="type [list, np.ndarray, pd.Series]", function_name="_check_input")
        else:
            for i in range(len(self.model_params)):
                if self.model_params[i].y_pred is None:
                    err.push('type_error', var_name="y_pred", given= "type None", expected="type [list, np.ndarray, pd.Series]", function_name="_check_input")

        #check for y_pred not None if model is uplift, if yes set to None as it is not required
        if self.model_params[0].model_type == 'uplift' and self.model_params[0].y_pred is not None:
            for i in range(len(self.model_params)):
                self.model_params[i].y_pred = None
        
        #check for y_prob is not None, if not None it cannot be an integer
        if self.model_params[0].y_prob is not None:
            if self.model_params[0].y_prob.dtype.kind == "i":
                err.push('type_error', var_name="y_prob", given= "type int", expected="type float", function_name="_check_input")

        #check for revenue and treatment_cost
        #only for uplift models based on expected profit perf metric        
        if self.model_params[0].model_type == 'uplift' and self.perf_metric_name == "expected_profit":
            exp_type = list((int, float))
            spl_range = (0, np.inf)
            #check if spl params are in expected type, otherwise throw exception
            for i in self.spl_params.keys() :
                if type(self.spl_params[i]) not in exp_type :
                    err.push('type_error', var_name=str(i), given=type(self.spl_params[i]), expected=exp_type, function_name="_check_input")
                #check if spl params are within expected range, otherwise throw exception
                try:
                    if type(self.spl_params[i]) != type(None):
                        if  self.spl_params[i] < spl_range[0]  or self.spl_params[i] > spl_range[1] :
                            err.push('value_error', var_name=str(i), given=self.spl_params[i],  expected="range " + str(spl_range), function_name="_check_input")
                except:
                    pass
            #check if in spl params, revenue value provided is not less than treatment_cost 
            try:
                if self.spl_params['revenue'] < self.spl_params['treatment_cost']:
                    err.push('value_error_compare', var_name_a="revenue", var_name_b="treatment_cost", function_name="_check_input")
            except:
                pass
        
        #print any exceptions occured        
        err.pop()
    
    def _get_confusion_matrix(self, y_true, y_pred, sample_weight, curr_p_var = None, feature_mask = None, **kwargs):

        """
        Compute confusion matrix

        Parameters
        ----------
        y_true : np.ndarray
                Ground truth target values.

        y_pred : np.ndarray
                Copy of predicted targets as returned by classifier.

        sample_weight : array of shape (n_samples,), default=None
                Used to normalize y_true & y_pred.

        curr_p_var : string, default=None
                Current protected variable

        feature_mask : dictionary of lists, default = None
                Stores the mask array for every protected variable applied on the x_test dataset.

        Returns
        -------
        Confusion matrix metrics based on privileged and unprivileged groups or a list of None if curr_p_var == None
        """
        #confusion matrix will only run for classification models
        if self._model_type_to_metric_lookup[self.model_params[0].model_type][0] == "classification" :
            if 'y_true' in kwargs:
                y_true = kwargs['y_true']

            
            if 'y_pred' in kwargs:
                y_pred = kwargs['y_pred']
            
            if curr_p_var is None :
                if y_pred is None:
                    return [None] * 4
                
                if sample_weight is None :
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                else :
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, sample_weight = sample_weight).ravel()
                return tp, fp, tn, fn 
            
            else:
                if y_pred is None:
                    return [None] * 8
                
                mask = feature_mask[curr_p_var]
                if sample_weight is None :
                    mask = self.feature_mask[curr_p_var]
                    tn_p, fp_p, fn_p, tp_p = confusion_matrix(np.array(y_true)[mask], np.array(y_pred)[mask]).ravel()
                    tn_u, fp_u, fn_u, tp_u  = confusion_matrix(np.array(y_true)[~mask], np.array(y_pred)[~mask]).ravel()
                else :
                    tn_p, fp_p, fn_p, tp_p = confusion_matrix(np.array(y_true)[mask], np.array(y_pred)[mask], sample_weight = sample_weight[mask]).ravel()
                    tn_u, fp_u, fn_u, tp_u  = confusion_matrix(np.array(y_true)[~mask], np.array(y_pred)[~mask], sample_weight = sample_weight[~mask]).ravel()
                return tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u
        else :
            if curr_p_var is None:
                return [None] * 4
    
            else:
                return [None] * 8
    
    def _select_fairness_metric_name(self):
        """
        Retrieves the fairness metric name based on the values of model_type, fair_concern, fair_impact and fair_priority.
        Name of the primary fairness metric to be used for computations in the evaluate() and/or compile() functions                
        """
        #if model type is uplift, will not use fairness_tree
        if self.fair_metric_name == 'auto':
            if self.model_params[0].model_type == 'uplift':
                self.fair_metric_name = 'rejected_harm'
            elif self.model_params[0].model_type == 'propensity':
                is_pos_label_favourable = True
                self._fairness_tree(is_pos_label_favourable)
            else:
                is_pos_label_favourable = False
                self._fairness_tree(is_pos_label_favourable)
        else :
            self.fair_metric_name

    def _get_e_lift(self, **kwargs):
        """
        Computes the empirical lift

        Other Parameters
        ----------
        y_pred_new : list of len = k of array of shape (n_samples,)
                Predicted targets as returned by classifier.

        Returns
        -----------
        e_lift : float or None
            Empirical lift value
        """
        #e_lift will only run for uplift models
        if self.model_params[0].model_type == 'uplift':
            
            y_train = self.model_params[1].y_train
            y_prob = self.model_params[1].y_prob
            
            if y_train is None :
                y_train = self.model_params[1].y_true
                
            if 'y_pred_new' in kwargs:
                y_prob = kwargs['y_pred_new']

                
            classes = np.array(['TR', 'TN', 'CR', 'CN'])
            p_base = np.array([np.mean(y_train == lab) for lab in classes])
            pC = p_base[2] + p_base[3]
            pT = p_base[0] + p_base[1]
            e_lift = (y_prob[:, 0] - y_prob[:, 1]) / pT \
                         + (y_prob[:, 3] - y_prob[:, 2]) / pC
            return e_lift
        else:
            return None
        
        
    def _compute_pred_outcome(self, **kwargs) :
        """
        Computes predicted outcome

        Other parameters
        ---------------
        y_pred_new : list of len = k of array of shape (n_samples,)
                Predicted targets as returned by classifier.

        Returns
        -----------
        pred_outcome : dictionary
                Contains the probabilities of the treatment and control groups for both rejection and acquiring
        """
        #pred_outcome will only run for uplift models
        if self.model_params[0].model_type == 'uplift':

            y_prob = [model.y_prob for model in self.model_params]
            y_train = [model.y_train  if model.y_train is not None else model.y_true for model in self.model_params]        
                        
            if 'y_pred_new' in kwargs:
                y_prob = kwargs['y_pred_new']

            if y_prob[0] is None or y_prob[1] is None:
                return None
            
            classes = np.array(['TR', 'TN', 'CR', 'CN'])
            model_alias = ['rej_', 'acq_']
            pred_outcome = {}
            
            for i in range(len(self.model_params)) :
                y_prob_temp = y_prob[i]
                y_train_temp = y_train[i]
                p_base = np.array([np.mean(y_train_temp == lab) for lab in classes])
                pC = p_base[2] + p_base[3]
                pT = p_base[0] + p_base[1]
                pOcT = y_prob_temp[:, 0] / pT
                pOcC = y_prob_temp[:, 2] / pC
                pred_outcome[model_alias[i] + 'treatment'] = pOcT
                pred_outcome[model_alias[i] + 'control'] = pOcC
            return pred_outcome

        else :
            return None

            