import numpy as np
from sklearn.metrics import confusion_matrix
from .fairness import Fairness
from ..metrics.fairness_metrics import FairnessMetrics
from ..metrics.performance_metrics import PerformanceMetrics
from ..metrics.newmetric import *
from ..util.utility import check_datatype, check_value
from ..config.constants import Constants
from ..util.errors import *

class CreditScoring(Fairness):
    """A class to evaluate and analyse fairness in credit scoring related applications.

    Class Attributes
    ------------------
    _model_type_to_metric_lookup: dictionary
                Used to associate the model type (key) with the metric type, expected size of positive and negative labels (value) & length of model_params respectively.
                e.g. {“rejection”: (“classification”, 2, 1), “uplift”: (“uplift”, 4, 1), “a_new_type”: (“regression”, -1, 1)}
    """

    _model_type_to_metric_lookup = {"credit": ("classification", 2, 1)}

    def __init__(self, model_params, fair_threshold, perf_metric_name = "balanced_acc", fair_metric_name = "auto", fair_concern = "eligible", fair_priority = "benefit", fair_impact = "normal", num_applicants = None, base_default_rate = None, fairness_metric_value_input = {}):
        """
        Parameters
        ----------
        model_params: list containing 1 ModelContainer object
                Data holder that contains all the attributes of the model to be assessed. Compulsory input for initialization. Single object corresponds to model_type of "default".

        fair_threshold: int or float
                Value between 0 and 100. If a float between 0 and 1 (not inclusive) is provided, it is converted to a percentage and the p % rule is used to calculate the fairness threshold value.        
                If an integer between 1 and 100 is provided, it is converted to a percentage and the p % rule is used to calculate the fairness threshold value.

        Instance Attributes
        --------------------
        perf_metric_name: string, default="balanced_acc"
                Name of the primary performance metric to be used for computations in the evaluate() and/or compile() functions.

        fair_metric_name : string, default="auto"
                Name of the primary fairness metric to be used for computations in the evaluate() and/or compile() functions.

        fair_concern: string, default="eligible"
                Used to specify a single fairness concern applied to all protected variables. Could be "eligible" or "inclusive" or "both".

        fair_priority: string, default="benefit"
                Used to pick the fairness metric according to the Fairness Tree methodology. Could be "benefit" or "harm"

        fair_impact: string, default="normal"
                Used to pick the fairness metric according to the Fairness Tree methodology. Could be "normal" or "significant" or "selective"

        num_applicants: dictionary of lists, default=None
                Contains the number of rejected applicants for the privileged and unprivileged groups for each protected feature.
                e.g. {"gender": [10, 20], "race": [12, 18]}

        base_default_rate: dictionary of lists, default=None
                Contains the base default rates for the privileged and unprivileged groups for each protected feature.
                e.g. {"gender": [10, 20], "race": [12, 18]} 

        fairness_metric_value_input : dictionary
                Contains the p_var and respective fairness_metric and value 
                e.g. {"gender": {"fnr_parity": 0.2}}

        _rejection_inference_flag: dictionary
                Flag to ascertain whether rejection inference technique should be used for each protected feature to impute the target value for rejected cases, allowing reject cohort to be used in model building.
                If both the base_default_rate & num_applicants are not None, the flag will be set to True.
                e.g. {"gender": True, "race": False, "age": True}

        _use_case_metrics: dictionary of lists
                Contains all the performance & fairness metrics for each use case.
                e.g. {"fair ": ["fnr_parity", ...], "perf": ["balanced_acc, ..."]}
                Dynamically assigned during initialisation by using the _metric_group_map in Fairness/Performance Metrics class and the _model_type_to_metric above.

        _input_validation_lookup: dictionary
                Contains the attribute and its correct data type for every argument passed by user. Used to perform the Utility checks.
                e.g. _input_validation_lookup = {
                "fair_threshold": [(float, int), (Constants().fair_threshold_low), Constants().fair_threshold_high],
                "fair_neutral_tolerance": [(float,),(Constants().fair_neutral_threshold_low), Constants().fair_neutral_threshold_high],
                "sample_weight": [(int,), (0, np.inf)],
                "perf_metric_name": [(str,), _use_case_metrics["perf"]],
                "fair_metric_name": [(str,), _use_case_metrics["fair"]],
                "concern": [(str,), ["eligible", "inclusion", "both"]]
                }

        spl_params : dictionary
                Dictionary of parameters that only belong to a use case

        k : int
                Integer from Constants class to calculate confidence interval

        array_size : int
                Integer from Constants class to fix array size

        decimals : int
                Integer from Constants class to fix number of decimals to round off

        err : object
                VeritasError object
        
        e_lift : float, default=None
                Empirical lift

        pred_outcome: dictionary, default=None
                Contains the probabilities of the treatment and control groups for both rejection and acquiring
        """
        super().__init__(model_params)
        self.perf_metric_name = perf_metric_name
        self.fair_threshold = fair_threshold
        self.fair_threshold_input = fair_threshold
        self.fair_neutral_tolerance = Constants().fair_neutral_tolerance 
        self.fair_concern = fair_concern
        self.fair_priority = fair_priority
        self.fair_impact = fair_impact
        self.fairness_metric_value_input = fairness_metric_value_input
        self.err = VeritasError()

        self._model_type_input()

        self.fair_metric_name = fair_metric_name
        self.fair_metric_input = fair_metric_name

        self._select_fairness_metric_name()
        self.check_perf_metric_name()
        self.check_fair_metric_name()

        self.spl_params = {'num_applicants':num_applicants, 'base_default_rate': base_default_rate}
                
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
        self.e_lift = None
        self.pred_outcome = None
        
        self._input_validation_lookup = {
            "fair_threshold": [(float, int), (Constants().fair_threshold_low), Constants().fair_threshold_high],
            "fair_neutral_tolerance": [(float,),(Constants().fair_neutral_threshold_low), Constants().fair_neutral_threshold_high],
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

        self._rejection_inference_flag = {}
        for var in self.model_params[0].p_var:
           self._rejection_inference_flag[var] = False

        if self.spl_params['base_default_rate'] is not None and self.spl_params['num_applicants'] is not None:
            for var in self.model_params[0].p_var:
                self._rejection_inference_flag[var] = True
        
        self._check_input()
        self._check_special_params()

    def _check_input(self):
        """
        Wrapper function to perform all checks using dictionaries of datatypes & dictionary of values.
        This function does not return any value. Instead, it raises an error when any of the checks from the Utility class fail.
        """
        #check datatype of input variables to ensure they are of the correct datatype
        check_datatype(self)

        #check datatype of input variables to ensure they are reasonable
        check_value(self)

        #check for model_params
        mp_given = len(self.model_params)
        mp_expected = self._model_type_to_metric_lookup[self.model_params[0].model_type][2]
        if mp_given != mp_expected:
            self.err.push('length_error', var_name="model_params", given=str(mp_given), expected=str(mp_expected), function_name="_check_input")

        #check for conflicting input values
        self._base_input_check()

        #check if input variables will the correct fair_metric_name based on fairness tree
        self._fairness_metric_value_input_check()

        # check if y_pred is not None 
        if self.model_params[0].y_pred is None:
            self.err.push('type_error', var_name="y_pred", given= "type None", expected="type [list, np.ndarray, pd.Series]", function_name="_check_input")

        # check if y_prob is float
        if self.model_params[0].y_prob is not None:
            if self.model_params[0].y_prob.dtype.kind == "i":
                self.err.push('type_error', var_name="y_prob", given= "type int", expected="type float", function_name="_check_input")

        #print any exceptions occured
        self.err.pop()

    def _check_special_params(self):
        """
        Perform data type and value checks for special params. 
        """
        #check that spl_params if provided contains dictionaries, otherwise throw exception
        for i in self.spl_params.keys() :
            if type(self.spl_params[i]) != dict and  type(self.spl_params[i]) != type(None):
                self._rejection_inference_flag = self._rejection_inference_flag.fromkeys(self._rejection_inference_flag, False)
                self.err.push('value_error', var_name=str(i), given=type(self.spl_params[i]), expected="dict", function_name="_check_special_params")
        
        #print any exceptions occured
        self.err.pop()

        #check that base_default_rate (under spl params) contains the protected variable under p_var, otherwise throw exception
        if self.spl_params['base_default_rate'] is not None:
            for var in self.model_params[0].p_var:
                if self._rejection_inference_flag[var] == True:
                    if var not in self.spl_params['base_default_rate'].keys():
                        self._rejection_inference_flag[var] = False 
                        self.err.push('value_error', var_name='base_default_rate', given="values for " + str(list(self.spl_params['base_default_rate'].keys())), expected="values for " + str( self.model_params[0].p_var), function_name="_check_special_params")
        
        #check that num_applicants (under spl params) contains the protected variable under p_var, otherwise throw exception        
        if self.spl_params['num_applicants'] is not None:
            for var in self.model_params[0].p_var:
                if self._rejection_inference_flag[var] == True:
                    if var not in self.spl_params['num_applicants'].keys():
                        self._rejection_inference_flag[var] = False 
                        self.err.push('value_error', var_name='num_applicants', given="values for " + str(list(self.spl_params['num_applicants'].keys())), expected="values for " + str(self.model_params[0].p_var), function_name="_check_special_params")
        #print any exceptions occured
        self.err.pop()

        #specify datatypes that are accepted for base_default_rate and num_applicants
        num_applicants_type = (int, float,)
        base_default_rate_type = (float,)
        #check that num_applicants contains correct datatype, otherwise throw exception
        if self.spl_params['num_applicants'] is not None:
            for i in self.spl_params["num_applicants"]:
                if self._rejection_inference_flag[i] == True :
                    for j in range(len(self.spl_params["num_applicants"].get(i))):
                        if type(self.spl_params["num_applicants"].get(i)[j]) in num_applicants_type:
                            if type(self.spl_params["num_applicants"].get(i)[j]) == float:
                                self.spl_params["num_applicants"].get(i)[j] = int(self.spl_params["num_applicants"].get(i)[j])
                        else:
                            self._rejection_inference_flag[i] = False 
                            self.err.push('type_error', var_name='num_applicants',
                                        given=str(type(self.spl_params["num_applicants"].get(i)[j])),
                                        expected=str(num_applicants_type), function_name="_check_special_params")
            
        #check that base_default_rate contains correct datatype, otherwise throw exception
        if self.spl_params['base_default_rate'] is not None:
            for i in self.spl_params["base_default_rate"]:
                if self._rejection_inference_flag[i] == True :
                    for j in range(len(self.spl_params["base_default_rate"].get(i))):
                        if type(self.spl_params["base_default_rate"].get(i)[j]) not in base_default_rate_type:
                            self._rejection_inference_flag[i] = False 
                            self.err.push('type_error', var_name='base_default_rate',
                                        given=str(type(self.spl_params["base_default_rate"].get(i)[j])),
                                        expected=str(base_default_rate_type), function_name="_check_special_params")

        #specify range of values that are accepted for base_default_rate and num_applicants
        num_applicants_range = (0, np.inf)
        base_default_rate_range = (0, 1)
        #check that num_applicants values are within range, otherwise throw exception
        if self.spl_params["num_applicants"] is not None:
            for k, l in self.spl_params['num_applicants'].items() :
                if self._rejection_inference_flag[k] == True :
                    for m in l :
                        if m <= 0:
                            self._rejection_inference_flag[k] = False 
                            self.err.push('value_error', var_name='num_applicants for ' + str(k), given=str(m),
                                          expected=str(num_applicants_range), function_name="_check_special_params")

        #check that base_default_rate values are within range, otherwise throw exception
        if self.spl_params["base_default_rate"] is not None:
            for k, l in self.spl_params['base_default_rate'].items() :
                if self._rejection_inference_flag[k] == True :
                    for m in l :
                        if  m < base_default_rate_range[0]  or m > base_default_rate_range[1] :
                            self._rejection_inference_flag[k] = False 
                            self.err.push('value_error', var_name='base_default_rate for ' + str(k), given=str(m),
                                        expected=str(base_default_rate_range), function_name="_check_special_params")

        #check for length of base_default_rate
        if self.spl_params['base_default_rate'] is not None:
            for var in self.spl_params['base_default_rate']:
                if self._rejection_inference_flag[var] == True :
                    if len(self.spl_params['base_default_rate'][var]) != 2:
                        self._rejection_inference_flag[var] = False 
                        self.err.push('length_error', var_name='base_default_rate', given=len(self.spl_params['base_default_rate'][var]), expected='2', function_name="_check_special_params")
        #check for length of num_applicants
        if self.spl_params['num_applicants'] is not None:
            for var in self.spl_params['num_applicants']:
                if self._rejection_inference_flag[var] == True :
                    if len(self.spl_params['num_applicants'][var]) != 2:
                        self._rejection_inference_flag[var] = False 
                        self.err.push('length_error', var_name='num_applicants', given=len(self.spl_params['num_applicants'][var]), expected='2', function_name="_check_special_params")
        
        #check for num of applicants if values in each index are consistent
        val_lst = []
        if self.spl_params['num_applicants'] is not None:
            for key, val in self.spl_params['num_applicants'].items():
                try:
                    val_lst += [sum(val)]
                except TypeError:
                    pass
            in_value = next(iter(self.spl_params['num_applicants'].items()))
            try:
                if sum(in_value[1]) != sum(val):
                    self._rejection_inference_flag = self._rejection_inference_flag.fromkeys(self._rejection_inference_flag, False)
                    self.err.push('conflict_error', var_name_a="num_applicants", some_string="inconsistent values",
                                      value=val_lst, function_name="_check_special_params")
                    
            except TypeError:
                pass

        #check for common base default rate based in spl params input values. If inconsistent, throw exception
        rejection_inference_filter = {k: v for k, v in self._rejection_inference_flag.items() if v == True}
        if len(rejection_inference_filter) > 0 :
            #Check for common base default rate
            check_cbdr = {}
            br_var = self.spl_params['base_default_rate']
            na_var = self.spl_params['num_applicants']
            if br_var is not None and na_var is not None:
                for i in self.model_params[0].p_var:
                    if self._rejection_inference_flag[i] == True :
                        for j in br_var:
                            for k in na_var:
                                if i == j and i == k:
                                    self.common_base_default_rate = (br_var.get(i)[0] * na_var.get(i)[0] + br_var.get(i)[1] * na_var.get(i)[1]) / (na_var.get(i)[0] + na_var.get(i)[1])
                                    check_cbdr[i] = self.common_base_default_rate
                br_value = next(iter(check_cbdr.items()))
                for val in check_cbdr:
                    if round(br_value[1], 5) != round(check_cbdr[val], 5):
                        self._rejection_inference_flag = self._rejection_inference_flag.fromkeys(self._rejection_inference_flag, False)
                        self.err.push('conflict_error', var_name_a="Common base default rates", some_string="inconsistent values", value=[round(br_value[1], 5), round(check_cbdr[val], 5)], function_name="_check_special_params")

        #check if num_applicants is more than length of y_true, otherwise throw exception
        na_var = self.spl_params['num_applicants']
        exp_out = np.array(self.model_params[0].y_true)
        if na_var is not None:
            for i in self.model_params[0].p_var:
                if self._rejection_inference_flag[i] == True :
                    idx = self.feature_mask[i]
                    pri_grp = self.model_params[0].p_grp.get(i)[0]
                    features = self.model_params[0].protected_features_cols[i].unique().tolist()
                    for j in features:
                        if j != pri_grp:
                            unpri_grp = j
        
                    if na_var.get(i)[0] < len(exp_out[idx]):
                        self._rejection_inference_flag[i] = False 
                        self.err.push('value_error_compare', var_name_a="Total number of applicants",
                                        var_name_b="total number of approvals", function_name="_check_special_params")
                    elif na_var.get(i)[1] < len(exp_out[~idx]):
                        self._rejection_inference_flag[i] = False 
                        self.err.push('value_error_compare', var_name_a="Total number of applicants",
                                        var_name_b="total number of approvals", function_name="_check_special_params")    
        
        #check if spl params provided are realistic, otherwise throw exception
        if self.model_params[0].y_pred is not None:
            tn, fp, fn, tp = self._get_confusion_matrix(self.model_params[0].y_true, self.model_params[0].y_pred, self.model_params[0].sample_weight)
            #check for acceptance cohort rate
            if fn < 0 or tn < 0:
                self._rejection_inference_flag = self._rejection_inference_flag.fromkeys(self._rejection_inference_flag, False)
                self.err.push('conflict_error', var_name_a="base_default_rate and/or num_applicants", some_string="unrealistic_input",
                                value='', function_name="_check_special_params")

            for curr_p_var in self.model_params[0].p_var :
                if self._rejection_inference_flag[curr_p_var] == True :
                    mask = self.feature_mask[curr_p_var]
                    tn_p, fp_p, fn_p, tp_p, tn_u, fp_u, fn_u, tp_u = self._get_confusion_matrix(y_true=self.model_params[0].y_true, y_pred=self.model_params[0].y_pred, sample_weight=self.model_params[0].sample_weight, curr_p_var = curr_p_var, feature_mask = self.feature_mask)
                    pri_grp = self.model_params[0].p_grp.get(curr_p_var)[0]
                    if self.spl_params['num_applicants'] is not None:
                        group_default_rate_p = fp_p / self.spl_params['num_applicants'].get(curr_p_var)[0]
                        group_default_rate_u = fp_u / self.spl_params['num_applicants'].get(curr_p_var)[1]
                        if group_default_rate_p > self.spl_params['base_default_rate'].get(curr_p_var)[0]:
                            self.err.push('conflict_error', var_name_a="base_default_rate", some_string="unrealistic_input",
                                            value='', function_name="_check_special_params")
                        if group_default_rate_u > self.spl_params['base_default_rate'].get(curr_p_var)[1]:
                            self.err.push('conflict_error', var_name_a="base_default_rate", some_string="unrealistic_input",
                                            value='', function_name="_check_special_params")
        #print any exceptions occured
        self.err.pop()

    def _select_fairness_metric_name(self):
        """
        Retrieves the fairness metric name based on the values of model_type, fair_concern, fair_impact, fair_priority.
        """
        err = VeritasError()
        if self.fair_metric_name == 'auto':
            self._fairness_tree()
        else :
            self.fair_metric_name

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
            
            rejection_inference_filter = {k: v for k, v in self._rejection_inference_flag.items() if v == True}

            if curr_p_var is None:
                if y_pred is None:
                    return [None] * 4

                if sample_weight is None or len(rejection_inference_filter) > 0:
                    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred).ravel()
                else :
                    tn, fp, fn, tp = confusion_matrix(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight).ravel()
                
                if len(rejection_inference_filter) > 0 :
    
                    M = self.spl_params['num_applicants'][list(rejection_inference_filter.keys())[0]][0] + self.spl_params['num_applicants'][list(rejection_inference_filter.keys())[0]][1]
                    fn = M * (1-self.common_base_default_rate) - tp
                    tn = M * (self.common_base_default_rate) - fp
                    
                return tp, fp, tn, fn
            else :
                if y_pred is None:
                    return [None] * 8

                mask = feature_mask[curr_p_var] 
                    
                if sample_weight is None or len(rejection_inference_filter) > 0 :
                    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true=np.array(y_true)[mask], y_pred=np.array(y_pred)[mask]).ravel()
                    tn_u, fp_u, fn_u, tp_u  = confusion_matrix(y_true=np.array(y_true)[~mask], y_pred=np.array(y_pred)[~mask]).ravel()
                else :
                    tn_p, fp_p, fn_p, tp_p = confusion_matrix(y_true=np.array(y_true)[mask], y_pred=np.array(y_pred)[mask], sample_weight = sample_weight[mask]).ravel()
                    tn_u, fp_u, fn_u, tp_u  = confusion_matrix(y_true=np.array(y_true)[~mask], y_pred=np.array(y_pred)[~mask], sample_weight = sample_weight[~mask]).ravel()
    
                if self._rejection_inference_flag[curr_p_var] == True :
                    fn_p = self.spl_params['num_applicants'][curr_p_var][0] * (1-self.spl_params['base_default_rate'][curr_p_var][0]) - tp_p
                    tn_p = self.spl_params['num_applicants'][curr_p_var][0] * self.spl_params['base_default_rate'][curr_p_var][0] - fp_p
                    fn_u= self.spl_params['num_applicants'][curr_p_var][1] * (1-self.spl_params['base_default_rate'][curr_p_var][1]) - tp_u
                    tn_u = self.spl_params['num_applicants'][curr_p_var][1] * self.spl_params['base_default_rate'][curr_p_var][1] - fp_u

                return tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u
        else :
            if curr_p_var is None :
                return [None] * 4  
            else :
                return [None] * 8