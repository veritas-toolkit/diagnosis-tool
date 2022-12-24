from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_percentage_error, roc_auc_score, log_loss
import numpy as np
from numpy import ma
import warnings
from scipy.stats import entropy 
from ..util.utility import *
from ..metrics import *
import concurrent.futures
from itertools import product

class FairnessMetrics:
    """
    Class that computes all the fairness metrics

    Class Attributes
    ----------
    map_fair_metric_to_group : dict
        Maps the fairness metrics to its name, metric_group (classification, uplift, or regression), type (parity or odds), whether the metric is related to tradeoff and whether the metric can be a primary metric.
        e.g. {'equal_opportunity': ('Equal Opportunity', 'classification', 'parity', True, True), 'equal_odds': ('Equalized Odds', 'classification', 'parity', True, True)}
    """
    map_fair_metric_to_group = {
        'disparate_impact': ('Disparate Impact', 'classification', 'ratio', True, True),
        'demographic_parity': ('Demographic Parity', 'classification', 'parity', True, True),
        'equal_opportunity': ('Equal Opportunity', 'classification', 'parity', True, True),
        'fpr_parity': ('False Positive Rate Parity', 'classification', 'parity', True, True),
        'tnr_parity': ('True Negative Rate Parity', 'classification', 'parity', True, True),
        'fnr_parity': ('False Negative Rate Parity', 'classification', 'parity', True, True),
        'ppv_parity': ('Positive Predictive Parity', 'classification', 'parity', True, True),
        'npv_parity': ('Negative Predictive Parity', 'classification', 'parity', True, True),
        'fdr_parity': ('False Discovery Rate Parity', 'classification', 'parity', True, True),
        'for_parity': ('False Omission Rate Parity', 'classification', 'parity', True, True),
        'equal_odds': ('Equalized Odds', 'classification', 'parity', True, True),
        'neg_equal_odds': ('Negative Equalized Odds', 'classification', 'parity', True, True),
        'calibration_by_group': ('Calibration by Group', 'classification', 'parity', True, True),
        'auc_parity': ('AUC Parity', 'classification', 'parity', False, True),
        'log_loss_parity': ('Log-loss Parity', 'classification', 'parity', False, True),
        'mi_independence': ('Mutual Information Independence', 'classification', 'information', False, False),
        'mi_separation': ('Mutual Information Separation', 'classification', 'information', False, False),
        'mi_sufficiency': ('Mutual Information Sufficiency', 'classification', 'information', False, False),
        'rmse_parity': ('Root Mean Squared Error Parity', 'regression', 'parity', False, True),
        'mape_parity': ('Mean Absolute Percentage Error Parity', 'regression', 'parity', False, True),
        'wape_parity': ('Weighted Absolute Percentage Error Parity', 'regression', 'parity', False, True),
        'rejected_harm': ('Harm from Rejection', 'uplift', 'parity', True, True),
        'acquire_benefit': ('Benefit from Acquiring', 'uplift', 'parity', False, True)
    }

    #to get cutomized metrics inherited from NewMetric class 
    for metric in newmetric.NewMetric.__subclasses__() :
        if metric.enable_flag ==True and metric.metric_type == "fair":
            map_fair_metric_to_group[metric.metric_name] =  (metric.metric_definition, metric.metric_group, metric.metric_parity_ratio, False, False)

    def __init__(self, use_case_object):
        """
        Parameters
        ------------------------
        use_case_object : object
                Object is initialised in use case classes.

        Instance Attributes
        ------------------------
        map_fair_metric_to_method : dict
                Maps the fairness metrics to the corresponding compute functions.
                e.g. {'equal_opportunity': _compute_equal_opportunity, 'equal_odds': _compute_equal_odds}

        result : dict of tuple, default=None
                Data holder that stores the following for every protected variable:
                - fairness metric value, corresponding confidence interval & neutral position for all fairness metrics.
                - feature distribution

        y_true : numpy.ndarray, default=None
                Ground truth target values.

        y_pred : numpy.ndarray, default=None
                Predicted targets as returned by classifier.

        y_train :numpy.ndarray, default=None
                Ground truth for training data.

        y_prob : numpy.ndarray, default=None
                Predicted probabilities as returned by classifier. 
                For uplift models, L = 4. Else, L = 1 where shape is (n_samples, L)

        feature_mask : numpy.ndarray, default=None
                Array of the masked protected variable according to the privileged and unprivileged groups.

        sample_weight : numpy.ndarray, default=None
                Used to normalize y_true & y_pred.

        p_var : list, default=None
                List of protected variables used for fairness analysis.

        fair_metric_name: str, default=None
                Name of the primary fairness metric to be used for computations in the evaluate() and/or compile() functions

        _use_case_metrics: dict of list, default=None
                Contains all the performance & fairness metrics for each use case. 
                {"fair ": ["fnr_parity", ...], "perf": ["balanced_accuracy, ..."]}
                Dynamically assigned during initialisation by using the _metric_group_map in Fairness/Performance Metrics class and the _model_type_to_metric above.
        """
        self.map_fair_metric_to_method = {
            'rmse_parity': self._compute_rmse_parity,
            'mape_parity': self._compute_mape_parity,
            'wape_parity': self._compute_wape_parity,
            'rejected_harm': self._compute_rejected_harm,
            'acquire_benefit': self._compute_benefit_from_acquiring
        }

        self.map_fair_metric_to_method_optimized = {
            'disparate_impact': self._compute_disparate_impact,
            'demographic_parity': self._compute_demographic_parity,
            'equal_opportunity': self._compute_equal_opportunity,
            'fpr_parity': self._compute_fpr_parity,
            'tnr_parity': self._compute_tnr_parity,
            'fnr_parity': self._compute_fnr_parity,
            'ppv_parity': self._compute_positive_predictive_parity,
            'npv_parity': self._compute_negative_predictive_parity,
            'fdr_parity': self._compute_false_discovery_rate_parity,
            'for_parity': self._compute_false_omission_rate_parity,
            'equal_odds': self._compute_equalized_odds,
            'neg_equal_odds': self._compute_negative_equalized_odds,
            'calibration_by_group': self._compute_calibration_by_group,
            'auc_parity': self._compute_auc_parity,
            'log_loss_parity': self._compute_log_loss_parity,
            'mi_independence': self._compute_mi_independence, 
            'mi_separation': self._compute_mi_separation, 
            'mi_sufficiency': self._compute_mi_sufficiency, 
        }
        
        #to get cutomized metrics inherited from NewMetric class
        for metric in NewMetric.__subclasses__() :
            if metric.enable_flag ==True and metric.metric_type == "fair":
                self.map_fair_metric_to_method[metric.metric_name] =  metric.compute
                self.map_fair_metric_to_group[metric.metric_name] =  (metric.metric_definition, metric.metric_group, metric.metric_parity_ratio, False, False)
                if metric.metric_name not in use_case_object._use_case_metrics["fair"]:
                    use_case_object._use_case_metrics["fair"].append(metric.metric_name)
                    
        self.result = {}
        self.y_true  = None
        self.y_prob  = None
        self.y_pred  = None
        self.feature_mask  = None
        self.p_var  = None
        self.sample_weight  = None
        self.fair_metric_name  = None
        self._use_case_metrics  = None
        self.use_case_object = use_case_object

    def execute_all_fair(self, n_threads, seed, eval_pbar):
        """
        Computes every fairness metric named inside the include_metrics list together with its associated confidence interval (dictionary), the privileged group metric value & the neutral position.

        Parameters
        ----------
        use_case_object : object
                A single initialized Fairness use case object (CreditScoring, CustomerMarketing, etc.)
        
        n_threads : int
                Number of currently active threads of a job

        seed : int
                Used to initialize the random number generator.

        eval_pbar : tqdm object
                Progress bar

        Returns
        ----------
        self.result: dict, default = None
                Data holder that stores the following for every protected variable.:
                - fairness metric value & corresponding confidence interval for all fairness metrics.
                - feature distribution
        """
        self.fair_metric_name = self.use_case_object.fair_metric_name
        self._use_case_metrics = self.use_case_object._use_case_metrics
        self.y_train = [model.y_train for model in self.use_case_object.model_params]
        self.p_var = [model.p_var for model in self.use_case_object.model_params]
        self.feature_mask = self.use_case_object.feature_mask
        self.curr_p_var = None
        self.result = {}

        #initialize result structure
        for i in self.p_var[0]:
            self.result[i] = {}
            idx = self.feature_mask[i]
            p_perc = sum(idx)/len(idx)
            feature_dist = { "privileged_group":p_perc, "unprivileged_group":1 - p_perc }  
            self.result[i]["feature_distribution"] = feature_dist
            self.result[i]["fair_metric_values"] = {}
            for j in self._use_case_metrics['fair']:
                if j in list(self.map_fair_metric_to_method.keys()) + list(self.map_fair_metric_to_method_optimized.keys()):
                    self.result[i]["fair_metric_values"][j] = [] 
        
        #update progress bar by 10
        eval_pbar.update(10)
        n = len(self.use_case_object.model_params[0].y_true)
        n_threads = check_multiprocessing(n_threads)
        
        #split k into k-1 times of random indexing compute and 1 time of original array compute
        if n_threads >= 1 and self.use_case_object.k > 1:
            #prepare k-1 arrays of random indices
            indices = []
            np.random.seed(seed)
            for ind in range(self.use_case_object.k-1):
                indices.append(np.random.choice(n, n, replace=True))

            #split the indices based on number threads and put into indexes, the size each indexes is the number of times each thread need to run
            indexes = []
            for i in range(n_threads):
                indexes.append([])
                for x in indices[i::n_threads]:
                    indexes[i].append(x)

            threads = []
            worker_progress = 36/n_threads
            with concurrent.futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
                #iterate through protected variables to drop one by one as part of leave-on-out
                for k in range(n_threads):
                    #deepcopy will be skipped if n_threads is 1
                    if n_threads == 1:
                        metric_obj = self
                    else:
                        metric_obj = deepcopy(self)
                    #submit each thread's work to thread pool
                    if len(indexes[k]) > 0:
                        threads.append(executor.submit(FairnessMetrics._execute_all_fair_map, metric_obj=metric_obj, index =indexes[k], eval_pbar=eval_pbar, worker_progress=worker_progress))
                        
                #retrive results from each thread
                for thread in threads:
                    mp_result = thread.result()
                    for i in self.p_var[0]:
                        for j in self._use_case_metrics['fair']:
                            if j in list(self.map_fair_metric_to_method.keys()) + list(self.map_fair_metric_to_method_optimized.keys()):
                                self.result[i]["fair_metric_values"][j] += mp_result[i]["fair_metric_values"][j]
        else:   
            #if multithreading is not triggered, directly update the progress bar by 36
            eval_pbar.update(36)

        #run 1 time of original array to compute fairness metrics
        FairnessMetrics._execute_all_fair_map(self, [np.arange(n)], eval_pbar, 1)
        
        #generate the final fairness metrics values and their CI based on k times of computation
        for i in self.p_var[0]:
            for j in self._use_case_metrics['fair']:
                if j in list(self.map_fair_metric_to_method.keys()) + list(self.map_fair_metric_to_method_optimized.keys()):
                    if self.result[i]["fair_metric_values"][j][-1][0] is None :
                        self.result[i]["fair_metric_values"][j] = (None, None, None)
                    else:
                        self.result[i]["fair_metric_values"][j] = self.result[i]['fair_metric_values'][j][-1] + (2*np.std([a_tuple[0] for a_tuple in self.result[i]["fair_metric_values"][j]]),)
        eval_pbar.update(6)

    def _execute_all_fair_map(metric_obj, index, eval_pbar, worker_progress):
        """
        Maps each thread's work for execute_all_fair()
        Parameters
        ----------
        metric_obj : FairnessMetrics object
        index : numpy.ndarray
        eval_pbar : tqdm object
                Progress bar
        worker_progress : int
                Progress bar progress for each thread
        """
        #get each iteration's progress in 2 decimals to update the progress bar
        prog = round(worker_progress/(len(index)), 2)

        #list to store all np arrays and combine for vectorization
        metric_obj.y_trues = []
        metric_obj.y_probs = []
        metric_obj.y_preds = []
        metric_obj.sample_weights = []
        metric_obj.feature_masks_list = []

        for idx in index:
            #prepare data
            metric_obj.y_true = [model.y_true[idx] for model in metric_obj.use_case_object.model_params]
            metric_obj.y_prob = [model.y_prob[idx] if model.y_prob is not None else None for model in metric_obj.use_case_object.model_params] 
            metric_obj.y_pred = [model.y_pred[idx] if model.y_pred is not None else None for model in metric_obj.use_case_object.model_params]
            metric_obj.sample_weight = [model.sample_weight[idx] if model.sample_weight is not None else None for model in metric_obj.use_case_object.model_params]
            metric_obj.e_lift = metric_obj.use_case_object.e_lift[idx] if metric_obj.use_case_object.e_lift is not None else None
            metric_obj.pred_outcome = {k: v[idx] for k, v in metric_obj.use_case_object.pred_outcome.items()} if metric_obj.use_case_object.pred_outcome is not None else {None}
            metric_obj.feature_mask = {k: v[idx] for k, v in metric_obj.use_case_object.feature_mask.items()}

            metric_obj.y_trues.append(metric_obj.y_true)
            metric_obj.y_probs.append(metric_obj.y_prob)
            metric_obj.y_preds.append(metric_obj.y_pred)
            metric_obj.feature_masks_list.append(metric_obj.feature_mask)
            metric_obj.sample_weights.append(metric_obj.sample_weight)

            for i in metric_obj.p_var[0]:
                metric_obj.curr_p_var = i     
                for j in metric_obj._use_case_metrics['fair']:
                    if j in metric_obj.map_fair_metric_to_method.keys():
                        metric_obj.result[i]["fair_metric_values"][j].append(metric_obj.map_fair_metric_to_method[j](obj=metric_obj))

        metric_obj.y_trues = np.array(metric_obj.y_trues)
        metric_obj.y_probs = np.array(metric_obj.y_probs)
        metric_obj.y_preds = np.array(metric_obj.y_preds)
        metric_obj.sample_weights = np.array(metric_obj.sample_weights)
        
        # Initialise entropy variables
        metric_obj.e_y_true = None
        metric_obj.e_y_pred = None
        metric_obj.e_y_true_y_pred = None

        metric_obj.feature_masks = {}
        for i in metric_obj.p_var[0]:

            metric_obj.curr_p_var = i
            metric_obj.feature_masks[i] = []

            metric_obj.e_curr_p_var = None
            metric_obj.e_joint = None
            metric_obj.e_y_true_curr_p_var = None
            metric_obj.e_y_pred_curr_p_var = None
            metric_obj.e_y_true_y_pred_curr_p_var = None

            for feature_mask in metric_obj.feature_masks_list:
                metric_obj.feature_masks[i].append((np.array(feature_mask[i])*1).reshape(1,1,-1)) #convert bool to int and reshape
            
            metric_obj.feature_masks[i] = np.concatenate(metric_obj.feature_masks[i])
            
            metric_obj.tp_ps, metric_obj.fp_ps, metric_obj.tn_ps, metric_obj.fn_ps, metric_obj.tp_us, metric_obj.fp_us, metric_obj.tn_us, metric_obj.fn_us = \
                metric_obj.use_case_object._get_confusion_matrix_optimized(
                        metric_obj.y_trues,
                        metric_obj.y_preds, 
                        metric_obj.sample_weights,
                        i, 
                        metric_obj.feature_masks
                    ) 

            for j in metric_obj._use_case_metrics['fair']:
                if j in metric_obj.map_fair_metric_to_method_optimized.keys():
                    metric_obj.result[i]["fair_metric_values"][j] += metric_obj.map_fair_metric_to_method_optimized[j](obj=metric_obj)
                    
        return metric_obj.result

    def translate_metric(self, metric_name, **kwargs ):
        """
        Computes the primary fairness metric value and its associate value for the privileged group, for the feature importance section.
        This function does not support rejection inference.

        Parameters
        ----------
        metric_name : str
                Name of fairness metric

        Other Parameters
        ----------
        kwargs : list

        Returns
        ----------
        result: dict, default = None
                Data holder that stores the following for every protected variable.:
                - fairness metric value, corresponding confidence interval for chosen fairness metric.
                - feature distribution
        """
        if metric_name in self.map_fair_metric_to_method.keys():
            return self.map_fair_metric_to_method[metric_name](**kwargs)
        if metric_name in self.map_fair_metric_to_method_optimized.keys():
            return self.map_fair_metric_to_method_optimized[metric_name](**kwargs)
        
    def _compute_disparate_impact(self, **kwargs):
        """
        Computes the ratio of approval rate between the privileged and unprivileged groups

        Returns
        ----------
        _compute_disparate_impact : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            pr_p = (tp_p + fp_p) / (tp_p + fp_p + tn_p + fn_p)
            pr_u = (tp_u + fp_u) / (tp_u + fp_u + tn_u + fn_u)
            return ((pr_p/pr_u)[0][0], pr_p[0][0])
        else:
            pr_p = (self.tp_ps + self.fp_ps) / (self.tp_ps + self.fp_ps + self.tn_ps + self.fn_ps)
            pr_u = (self.tp_us + self.fp_us) / (self.tp_us + self.fp_us + self.tn_us + self.fn_us)
            return list(map(tuple, np.stack((pr_p/pr_u, pr_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_demographic_parity(self, **kwargs):
        """
        Computes the difference in approval rate between the privileged and unprivileged groups

        Returns
        ----------
        _compute_demographic_parity : list of tuple of floats
                Fairness metric value and privileged group metric value

        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            pr_p = (tp_p + fp_p) / (tp_p + fp_p + tn_p + fn_p)
            pr_u = (tp_u + fp_u) / (tp_u + fp_u + tn_u + fn_u)
            return ((pr_p - pr_u)[0][0], pr_p[0][0])
        else:
            pr_p = (self.tp_ps + self.fp_ps) / (self.tp_ps + self.fp_ps + self.tn_ps + self.fn_ps)
            pr_u = (self.tp_us + self.fp_us) / (self.tp_us + self.fp_us + self.tn_us + self.fn_us)
            return list(map(tuple, np.stack((pr_p - pr_u, pr_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_false_omission_rate_parity(self, **kwargs):
        """
        Computes the difference in negative predictive values between the privileged and unprivileged groups

        Returns
        ----------
        _compute_false_omission_rate_parity : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            for_p = fn_p / (tn_p + fn_p)
            for_u = fn_u / (tn_u + fn_u)
            return ((for_p - for_u)[0][0], for_p[0][0])
        else:
            for_p = self.fn_ps / (self.tn_ps + self.fn_ps)
            for_u = self.fn_us / (self.tn_us + self.fn_us)
            return list(map(tuple, np.stack((for_p - for_u, for_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_false_discovery_rate_parity(self, **kwargs):
        """
        Computes the difference in false discovery rate values between the privileged and unprivileged groups

        Returns
        ----------
        _compute_false_discovery_rate_parity : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            fdr_p = fp_p / (tp_p + fp_p)
            fdr_u = fp_u / (tp_u + fp_u)
            return ((fdr_p - fdr_u)[0][0], fdr_p[0][0])
        else:
            fdr_p = self.fp_ps / (self.tp_ps + self.fp_ps)
            fdr_u = self.fp_us / (self.tp_us + self.fp_us)
            return list(map(tuple, np.stack((fdr_p - fdr_u, fdr_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_positive_predictive_parity(self, **kwargs):
        """
        Computes the difference in positive predictive values between the privileged and unprivileged groups
        
        Returns
        ----------
        _compute_positive_predictive_parity : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            ppv_p = tp_p / (tp_p + fp_p)
            ppv_u = tp_u / (tp_u + fp_u)
            return ((ppv_p - ppv_u)[0][0], ppv_p[0][0])
        else:
            ppv_p = self.tp_ps / (self.tp_ps + self.fp_ps)
            ppv_u = self.tp_us / (self.tp_us + self.fp_us)
            return list(map(tuple, np.stack((ppv_p - ppv_u, ppv_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_negative_predictive_parity(self, **kwargs):
        """
        Computes the difference in negative predictive values between the privileged and unprivileged groups

        Returns
        ----------
        _compute_negative_predictive_parity : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            npv_p = tn_p / (tn_p + fn_p)
            npv_u = tn_u / (tn_u + fn_u)
            return ((npv_p - npv_u)[0][0], npv_p[0][0])
        else:
            npv_p = self.tn_ps / (self.tn_ps + self.fn_ps)
            npv_u = self.tn_us / (self.tn_us + self.fn_us)
            return list(map(tuple, np.stack((npv_p - npv_u, npv_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_fnr_parity(self, **kwargs):
        """
        Computes the difference in false negative rates between the privileged and unprivileged groups

        Returns
        ----------
        _compute_fnr_parity : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            fnr_p = fn_p / (tp_p + fn_p)
            fnr_u = fn_u / (tp_u + fn_u)
            return ((fnr_p - fnr_u)[0][0], fnr_p[0][0])
        else:
            fnr_p = self.fn_ps / (self.tp_ps + self.fn_ps)
            fnr_u = self.fn_us / (self.tp_us + self.fn_us)
            return list(map(tuple, np.stack((fnr_p - fnr_u, fnr_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_fpr_parity(self, **kwargs): 
        """
        Computes the difference in false positive rates between the privileged and unprivileged groups

        Returns
        ----------
        _compute_fpr_parity : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            fpr_p = fp_p / (tn_p + fp_p)
            fpr_u = fp_u / (tn_u + fp_u)
            return ((fpr_p - fpr_u)[0][0], fpr_p[0][0])
        else:
            fpr_p = self.fp_ps / (self.tn_ps + self.fp_ps)
            fpr_u = self.fp_us / (self.tn_us + self.fp_us)
            return list(map(tuple, np.stack((fpr_p - fpr_u, fpr_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_tnr_parity(self, **kwargs):
        """
        Computes the difference in false negative rates between the privileged and unprivileged groups

        Returns
        ----------
        _compute_tnr_parity : list tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            tnr_p = tn_p / (tn_p + fp_p)
            tnr_u = tn_u / (tn_u + fp_u)
            return ((tnr_p - tnr_u)[0][0], tnr_p[0][0])
        else:
            tnr_p = self.tn_ps / (self.tn_ps + self.fp_ps)
            tnr_u = self.tn_us / (self.tn_us + self.fp_us)
            return list(map(tuple, np.stack((tnr_p - tnr_u, tnr_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_equalized_odds(self, **kwargs):
        """
        Computes the equalized odds
        
        Returns
        ----------
        _compute_equalized_odds : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            tpr_p = tp_p / (tp_p + fn_p)
            tpr_u = tp_u / (tp_u + fn_u)
            fpr_p = fp_p / (fp_p + tn_p)
            fpr_u = fp_u / (fp_u + tn_u)
            return ((((tpr_p - tpr_u) + (fpr_p - fpr_u))/2)[0][0], ((tpr_p + fpr_p)/2)[0][0])
        else:
            tpr_p = self.tp_ps / (self.tp_ps + self.fn_ps)
            tpr_u = self.tp_us / (self.tp_us + self.fn_us)
            fpr_p = self.fp_ps / (self.fp_ps + self.tn_ps)
            fpr_u = self.fp_us / (self.fp_us + self.tn_us)
            return list(map(tuple, np.stack((((tpr_p - tpr_u) + (fpr_p - fpr_u))/2, (tpr_p + fpr_p)/2), axis=1).reshape(-1, 2).tolist()))

    def _compute_negative_equalized_odds(self, **kwargs):
        """
        Computes the negative equalized odds

        Returns
        ----------
        _compute_negative_equalized_odds : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            tnr_p = tn_p / (tn_p + fp_p)
            tnr_u = tn_u / (tn_u + fp_u)
            fnr_p = fn_p / (fn_p + tp_p)
            fnr_u = fn_u / (fn_u + tp_u)
            return ((((tnr_p - tnr_u) + (fnr_p - fnr_u)) / 2)[0][0], ((tnr_p + fnr_p) / 2)[0][0])
        else:
            tnr_p = self.tn_ps / (self.tn_ps + self.fp_ps)
            tnr_u = self.tn_us / (self.tn_us + self.fp_us)
            fnr_p = self.fn_ps / (self.fn_ps + self.tp_ps)
            fnr_u = self.fn_us / (self.fn_us + self.tp_us)
            return list(map(tuple, np.stack((((tnr_p - tnr_u) + (fnr_p - fnr_u)) / 2, (tnr_p + fnr_p) / 2), axis=1).reshape(-1, 2).tolist()))

    def _compute_rmse_parity(self, **kwargs):
        """
        Computes the difference in root mean squared error between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_rmse_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        mask = self.feature_mask[self.curr_p_var]
        y_true = self.y_true[0]
        y_pred = self.y_pred[0]

        if 'y_pred_new' in kwargs:
            y_pred=kwargs['y_pred_new'][0]
        
        if self.sample_weight[0] is not None: 
            sample_weight_p = np.array(self.sample_weight[0])[mask]
            sample_weight_u = np.array(self.sample_weight[0])[~mask]
        else:
            sample_weight_p = None
            sample_weight_u = None
        
        rmse_p = mean_squared_error(y_true=np.array(y_true)[mask], y_pred=np.array(y_pred)[mask], sample_weight=sample_weight_p) ** 0.5
        rmse_u = mean_squared_error(y_true=np.array(y_true)[~mask], y_pred=np.array(y_pred)[~mask], sample_weight=sample_weight_u) ** 0.5
        return (rmse_p - rmse_u, rmse_p)

    def _compute_mape_parity(self, **kwargs):
        """
        Computes the difference in mean average percentage error between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_mape_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        mask = self.feature_mask[self.curr_p_var]
        y_true = self.y_true[0]
        y_pred = self.y_pred[0]

        if 'y_pred_new' in kwargs:
            y_pred=kwargs['y_pred_new'][0]
            
        if self.sample_weight[0] is not None: 
            sample_weight_p = np.array(self.sample_weight[0])[mask]
            sample_weight_u = np.array(self.sample_weight[0])[~mask]
        else:
            sample_weight_p = None
            sample_weight_u = None

        mape_p = mean_absolute_percentage_error(y_true=np.array(y_true)[mask], y_pred=np.array(y_pred)[mask], sample_weight=sample_weight_p)
        mape_u = mean_absolute_percentage_error(y_true=np.array(y_true)[~mask], y_pred=np.array(y_pred)[~mask], sample_weight=sample_weight_u)

        return (mape_p - mape_u, mape_p)

    def _compute_wape_parity(self, **kwargs):
        """
        Computes the difference in weighted average percentage error between the privileged and unprivileged groups
    
        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
            Copy of predicted targets as returned by classifier.
    
        Returns
        ----------
        _compute_wape_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        mask = self.feature_mask[self.curr_p_var]
        y_true = self.y_true[0]
        y_pred = self.y_pred[0]
    
        if 'y_pred_new' in kwargs:
            y_pred=kwargs['y_pred_new'][0]
    
        wape_p = np.sum(np.absolute(np.subtract(y_true[mask], y_pred[mask])))/ np.sum(y_true[mask])
        wape_u = np.sum(np.absolute(np.subtract(y_true[~mask], y_pred[~mask])))/ np.sum(y_true[~mask])
            
        return (wape_p - wape_u, wape_p)

    def _compute_log_loss_parity(self, **kwargs):
        """
        Computes the difference in logistic loss or cross entropy loss between the privileged and unprivileged groups

        Returns
        ----------
        _compute_log_loss_parity : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            mask = self.feature_mask[self.curr_p_var]
            y_true = self.y_true[0]
            y_prob=kwargs['y_pred_new'][0]
            if self.sample_weight[0] is not None: 
                sample_weight_p = np.array(self.sample_weight[0])[mask]
                sample_weight_u = np.array(self.sample_weight[0])[~mask]
            else:
                sample_weight_p = None
                sample_weight_u = None   
            log_loss_p = log_loss(y_true=np.array(y_true)[mask], y_pred=np.array(y_prob)[mask], sample_weight = sample_weight_p)
            log_loss_u = log_loss(y_true=np.array(y_true)[~mask], y_pred=np.array(y_prob)[~mask], sample_weight = sample_weight_u)
            return (log_loss_p - log_loss_u, log_loss_p)
        else:
            mask = self.feature_masks[self.curr_p_var]
            log_loss_score = -(self.y_trues*ma.log(self.y_probs) + (1-self.y_trues)*ma.log(1-self.y_probs))
            log_loss_p = np.sum(log_loss_score*mask, 2)/np.sum(mask, 2)
            log_loss_u = np.sum(log_loss_score*(1-mask), 2)/np.sum((1-mask), 2)
            return list(map(tuple, np.stack((log_loss_p - log_loss_u, log_loss_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_auc_parity(self, **kwargs):
        """
        Computes the difference in area under roc curve between the privileged and unprivileged groups

        Returns
        ----------
        _compute_auc_parity : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            mask = self.feature_mask[self.curr_p_var]
            y_true = self.y_true[0]
            y_prob=kwargs['y_pred_new'][0]
            if self.sample_weight[0] is not None: 
                sample_weight_p = np.array(self.sample_weight[0])[mask]
                sample_weight_u = np.array(self.sample_weight[0])[~mask]
            else:
                sample_weight_p = None
                sample_weight_u = None
            roc_auc_score_p = roc_auc_score(y_true=np.array(y_true)[mask], y_score=np.array(y_prob)[mask], sample_weight=sample_weight_p)
            roc_auc_score_u = roc_auc_score(y_true=np.array(y_true)[~mask], y_score=np.array(y_prob)[~mask], sample_weight=sample_weight_u)
            return (roc_auc_score_p - roc_auc_score_u, roc_auc_score_p)
        else:
            mask = self.feature_masks[self.curr_p_var]
            idx = self.y_probs.argsort(axis=2)[:,:,::-1] # sort by descending order
            y_probs = np.take_along_axis(self.y_probs, idx, axis=2)
            y_trues = np.take_along_axis(self.y_trues, idx, axis=2)
            mask = np.take_along_axis(mask, idx, axis=2)

            TPR_p = np.cumsum(y_trues*mask, axis=2)/np.sum(y_trues*mask, axis=2, keepdims=True)
            FPR_p = np.cumsum((1-y_trues)*mask, axis=2)/np.sum((1-y_trues)*mask, axis=2, keepdims=True)
            TPR_p = np.append(np.zeros((TPR_p.shape[0],TPR_p.shape[1],1)), TPR_p, axis=2) # append starting point (0)
            FPR_p = np.append(np.zeros((FPR_p.shape[0],FPR_p.shape[1],1)), FPR_p, axis=2)
            auc_p = np.trapz(TPR_p, FPR_p, axis=2)

            TPR_u = np.cumsum(y_trues*(1-mask), axis=2)/np.sum(y_trues*(1-mask), axis=2, keepdims=True)
            FPR_u = np.cumsum((1-y_trues)*(1-mask), axis=2)/np.sum((1-y_trues)*(1-mask), axis=2, keepdims=True)
            TPR_u = np.append(np.zeros((TPR_u.shape[0],TPR_u.shape[1],1)), TPR_u, axis=2) # append starting point (0)
            FPR_u = np.append(np.zeros((FPR_u.shape[0],FPR_u.shape[1],1)), FPR_u, axis=2)
            auc_u = np.trapz(TPR_u, FPR_u, axis=2)
            return list(map(tuple, np.stack((auc_p - auc_u, auc_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_equal_opportunity(self, **kwargs):
        """
        Computes the equal opportunity

        Returns
        ----------
        _compute_equal_opportunity : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            tpr_p = tp_p / (tp_p + fn_p)
            tpr_u = tp_u / (tp_u + fn_u)
            return ((tpr_p - tpr_u)[0][0], tpr_p[0][0])
        else:
            tpr_p = self.tp_ps / (self.tp_ps + self.fn_ps)
            tpr_u = self.tp_us / (self.tp_us + self.fn_us)
            return list(map(tuple, np.stack((tpr_p - tpr_u, tpr_p), axis=1).reshape(-1, 2).tolist()))

    def _compute_calibration_by_group(self, **kwargs):
        """
        Computes the calibration by group within protected variable

        Returns
        ----------
        _compute_calibration_by_group : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            feature_mask = {k: np.array(v*1).reshape(1, 1, -1) for k, v in self.feature_mask.items()}
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1),
                curr_p_var=self.curr_p_var, 
                feature_mask=feature_mask,
            )
            ppv_p = tp_p / (tp_p + fp_p)
            ppv_u = tp_u / (tp_u + fp_u)
            for_p = fn_p / (tn_p + fn_p)
            for_u = fn_u / (tn_u + fn_u)
            return ((((ppv_p - ppv_u) + (for_p - for_u)) / 2)[0][0], ((ppv_p - ppv_u) / 2)[0][0])
        else:
            ppv_p = self.tp_ps / (self.tp_ps + self.fp_ps)
            ppv_u = self.tp_us / (self.tp_us + self.fp_us)
            for_p = self.fn_ps / (self.tn_ps + self.fn_ps)
            for_u = self.fn_us / (self.tn_us + self.fn_us)  
            return list(map(tuple, np.stack((((ppv_p - ppv_u) + (for_p - for_u)) / 2, (ppv_p - ppv_u) / 2), axis=1).reshape(-1, 2).tolist()))

    def _compute_mi_independence(self, **kwargs) :
        """
        Compute Mutual Information independence

        Returns
        ----------
        _compute_mi_independence : list of tuple of floats
                Fairness metric value and privileged group metric value
        """ 
        if 'y_pred_new' in kwargs:
            mask = self.feature_mask[self.curr_p_var]
            y_pred=kwargs['y_pred_new'][0]
            df = pd.DataFrame({'y_pred': y_pred, 'curr_p_var': mask})
            e_y_pred = self._get_entropy(df,['y_pred'])
            e_curr_p_var = self._get_entropy(df,['curr_p_var'])
            e_joint = self._get_entropy(df,['y_pred', 'curr_p_var'])
            mi_independence = (e_y_pred + e_curr_p_var - e_joint)/e_curr_p_var
            return (mi_independence, None)
    
        else:
            mask = self.feature_masks[self.curr_p_var]
            if self.e_y_pred is None:
                proportion_pred = [
                    np.sum(self.y_preds, 2)/self.y_preds.shape[2],
                    np.sum(1-self.y_preds, 2)/self.y_preds.shape[2],
                ]
                proportion_pred = np.stack(proportion_pred, axis=2)
                e_y_pred = -np.sum(proportion_pred*ma.log(proportion_pred), axis=2) 
                self.e_y_pred = e_y_pred    
            else:
                e_y_pred = self.e_y_pred

            if self.e_curr_p_var is None:    
                proportion_p_var = [
                    np.sum(mask, 2)/mask.shape[2],
                    np.sum(1-mask, 2)/mask.shape[2]
                ]
                proportion_p_var = np.stack(proportion_p_var, axis=2)
                e_curr_p_var = -np.sum(proportion_p_var*ma.log(proportion_p_var), axis=2) 
                self.e_curr_p_var = e_curr_p_var
            else:
                e_curr_p_var = self.e_curr_p_var

            if self.e_joint is None:
                proportion_join = []
                cart_product = product([self.y_preds, 1-self.y_preds], [mask, 1-mask])
                for i in cart_product:
                    p = i[0]*i[1]
                    proportion_join.append(
                        np.sum(p, 2)/p.shape[2]
                    )    
                proportion_join = np.stack(proportion_join, axis=2)
                e_joint = -np.sum(proportion_join*ma.log(proportion_join), axis=2) 
                self.e_joint = e_joint
            else:
                e_joint = self.e_joint

            mi_independence = (e_y_pred + e_curr_p_var - e_joint)/e_curr_p_var
            mi_independence = mi_independence.reshape(-1).tolist()

            return [(v, None) for v in mi_independence]

    def _compute_mi_separation(self, **kwargs) :
        """
        Compute Mutual Information separation

        Returns
        ----------
        _compute_mi_separation : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            mask = self.feature_mask[self.curr_p_var]
            y_pred=kwargs['y_pred_new'][0]
            df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'curr_p_var': mask})
            e_y_true_curr_p_var = self._get_entropy(df,['y_true', 'curr_p_var'])
            e_y_pred_curr_p_var = self._get_entropy(df,['y_pred', 'curr_p_var'])
            e_y_true_y_pred_curr_p_var = self._get_entropy(df,['y_true', 'y_pred', 'curr_p_var'])
            e_y_true = self._get_entropy(df,['y_true'])
            e_curr_p_var_y_true_conditional = e_y_true_curr_p_var - e_y_true
            mi_separation = (e_y_true_curr_p_var + e_y_pred_curr_p_var - e_y_true_y_pred_curr_p_var - e_y_true)/e_curr_p_var_y_true_conditional
            return (mi_separation, None)   
        
        else:
            mask = self.feature_masks[self.curr_p_var]
            if self.e_y_true is None:
                proportion_true = [
                    np.sum(self.y_trues, 2)/self.y_trues.shape[2],
                    np.sum(1-self.y_trues, 2)/self.y_trues.shape[2],
                ]
                proportion_true = np.stack(proportion_true, axis=2)
                e_y_true = -np.sum(proportion_true*ma.log(proportion_true), axis=2) 
                self.e_y_true = e_y_true    
            else:
                e_y_true = self.e_y_true

            if self.e_y_true_curr_p_var is None:
                proportion_join = []
                cart_product = product([self.y_trues, 1-self.y_trues], [mask, 1-mask])
                for i in cart_product:
                    p = i[0]*i[1]
                    proportion_join.append(
                        np.sum(p, 2)/p.shape[2]
                    )    
                proportion_join = np.stack(proportion_join, axis=2)
                e_y_true_curr_p_var = -np.sum(proportion_join*ma.log(proportion_join), axis=2) 
                self.e_y_true_curr_p_var = e_y_true_curr_p_var
            else:
                e_y_true_curr_p_var = self.e_y_true_curr_p_var

            if self.e_y_pred_curr_p_var is None:
                proportion_join = []
                cart_product = product([self.y_preds, 1-self.y_preds], [mask, 1-mask])
                for i in cart_product:
                    p = i[0]*i[1]
                    proportion_join.append(
                        np.sum(p, 2)/p.shape[2]
                    )    
                proportion_join = np.stack(proportion_join, axis=2)
                e_y_pred_curr_p_var = -np.sum(proportion_join*ma.log(proportion_join), axis=2) 
                self.e_y_pred_curr_p_var = e_y_pred_curr_p_var
            else:
                e_y_pred_curr_p_var = self.e_y_pred_curr_p_var

            if self.e_y_true_y_pred_curr_p_var is None:
                proportion_join = []
                cart_product = product([self.y_trues, 1-self.y_trues], [self.y_preds, 1-self.y_preds], [mask, 1-mask])
                for i in cart_product:
                    p = i[0]*i[1]*i[2]
                    proportion_join.append(
                        np.sum(p, 2)/p.shape[2]
                    )    
                proportion_join = np.stack(proportion_join, axis=2)
                e_y_true_y_pred_curr_p_var = -np.sum(proportion_join*ma.log(proportion_join), axis=2) 
                self.e_y_true_y_pred_curr_p_var = e_y_true_y_pred_curr_p_var
            else:
                e_y_true_y_pred_curr_p_var = self.e_y_true_y_pred_curr_p_var

            e_curr_p_var_y_true_conditional = e_y_true_curr_p_var - e_y_true
            mi_separation = (e_y_true_curr_p_var + e_y_pred_curr_p_var - e_y_true_y_pred_curr_p_var - e_y_true)/e_curr_p_var_y_true_conditional
            mi_separation = mi_separation.reshape(-1).tolist()

            return [(v, None) for v in mi_separation]

    def _compute_mi_sufficiency(self, **kwargs) :
        """
        Compute Mutual Information sufficiency

        Returns
        ----------
        _compute_mi_sufficiency : list of tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            mask = self.feature_mask[self.curr_p_var]
            y_pred=kwargs['y_pred_new'][0]
            df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'curr_p_var': mask})
            e_y_pred_curr_p_var = self._get_entropy(df,['y_pred', 'curr_p_var'])
            e_y_true_y_pred = self._get_entropy(df,['y_true', 'y_pred'])
            e_y_true_y_pred_curr_p_var = self._get_entropy(df,['y_true', 'y_pred', 'curr_p_var'])
            e_y_pred = self._get_entropy(df,['y_pred'])
            e_curr_p_var_y_pred_conditional = e_y_pred_curr_p_var - e_y_pred
            mi_sufficiency = (e_y_pred_curr_p_var + e_y_true_y_pred - e_y_true_y_pred_curr_p_var - e_y_pred)/e_curr_p_var_y_pred_conditional
            return (mi_sufficiency, None)
        
        else:
            mask = self.feature_masks[self.curr_p_var]
            if self.e_y_pred is None:
                proportion_pred = [
                    np.sum(self.y_preds, 2)/self.y_preds.shape[2],
                    np.sum(1-self.y_preds, 2)/self.y_preds.shape[2],
                ]
                proportion_pred = np.stack(proportion_pred, axis=2)
                e_y_pred = -np.sum(proportion_pred*ma.log(proportion_pred), axis=2) 
                self.e_y_pred = e_y_pred    
            else:
                e_y_pred = self.e_y_pred

            if self.e_y_true_y_pred is None:
                proportion_join = []
                cart_product = product([self.y_trues, 1-self.y_trues], [self.y_preds, 1-self.y_preds])
                for i in cart_product:
                    p = i[0]*i[1]
                    proportion_join.append(
                        np.sum(p, 2)/p.shape[2]
                    )    
                proportion_join = np.stack(proportion_join, axis=2)
                e_y_true_y_pred = -np.sum(proportion_join*ma.log(proportion_join), axis=2) 
                self.e_y_true_y_pred = e_y_true_y_pred
            else:
                e_y_true_y_pred = self.e_y_true_y_pred

            if self.e_y_pred_curr_p_var is None:
                proportion_join = []
                cart_product = product([self.y_preds, 1-self.y_preds], [mask, 1-mask])
                for i in cart_product:
                    p = i[0]*i[1]
                    proportion_join.append(
                        np.sum(p, 2)/p.shape[2]
                    )    
                proportion_join = np.stack(proportion_join, axis=2)
                e_y_pred_curr_p_var = -np.sum(proportion_join*ma.log(proportion_join), axis=2) 
                self.e_y_pred_curr_p_var = e_y_pred_curr_p_var
            else:
                e_y_pred_curr_p_var = self.e_y_pred_curr_p_var

            if self.e_y_true_y_pred_curr_p_var is None:
                proportion_join = []
                cart_product = product([self.y_trues, 1-self.y_trues], [self.y_preds, 1-self.y_preds], [mask, 1-mask])
                for i in cart_product:
                    p = i[0]*i[1]*i[2]
                    proportion_join.append(
                        np.sum(p, 2)/p.shape[2]
                    )    
                proportion_join = np.stack(proportion_join, axis=2)
                e_y_true_y_pred_curr_p_var = -np.sum(proportion_join*ma.log(proportion_join), axis=2) 
                self.e_y_true_y_pred_curr_p_var = e_y_true_y_pred_curr_p_var
            else:
                e_y_true_y_pred_curr_p_var = self.e_y_true_y_pred_curr_p_var

            e_curr_p_var_y_pred_conditional = e_y_pred_curr_p_var - e_y_pred
            mi_sufficiency = (e_y_pred_curr_p_var + e_y_true_y_pred - e_y_true_y_pred_curr_p_var - e_y_pred)/e_curr_p_var_y_pred_conditional
            mi_sufficiency = mi_sufficiency.reshape(-1).tolist()

            return [(v, None) for v in mi_sufficiency]

    def _get_entropy(self, df, columns) :
        """
        Compute the entropy

        Parameters
        -----------
        df : pandas.DataFrame
                Data set

        columns : list of strings
                Column names

        Returns
        -----------
        entropy_calc : float
                Entropy value
        """
        probabilities = (df.groupby(columns).size().reset_index(name='probability')['probability']/df.shape[0]).values[0:]
        entropy_calc = entropy(probabilities)
        return entropy_calc
    
    def _compute_rejected_harm(self, selection_threshold = None, **kwargs):

        """
        The Rejection Empirical Lift is Empirical Lift of the Marketing Rejection Uplift Model
        Computes the difference in rejection rates between treatment and control groups

        Parameters
        -----------
        selection_threshold : float, default = None

        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_rejected_harm : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if selection_threshold is None:
            selection_threshold = self.use_case_object.selection_threshold

        mask_list = self.feature_mask[self.curr_p_var]
        e_lift =  self.e_lift
        pred_outcome = self.pred_outcome
        
        if pred_outcome is None or e_lift is None:
            return (None, None)
        
        if 'y_pred_new' in kwargs:
            y_prob=kwargs['y_pred_new']
            e_lift = self.use_case_object._get_e_lift(y_pred_new=y_prob[1])
            pred_outcome = self.use_case_object._compute_pred_outcome(y_pred_new=y_prob)

        def _rej_harm(pred_outcome, selection_threshold, e_lift, mask_list) :
            bools = np.array([i > selection_threshold for i in e_lift])[mask_list]
            pRcT = pred_outcome['rej_treatment'][(mask_list)][(bools)]
            pRcC = pred_outcome['rej_control'][(mask_list)][(bools)]
            reject_harm = sum(pRcT - pRcC) / len(bools)
            return reject_harm

        rej_harm_p = _rej_harm(pred_outcome, selection_threshold, e_lift, mask_list)
        rej_harm_u = _rej_harm(pred_outcome, selection_threshold, e_lift, ~mask_list)

        return ((rej_harm_p - rej_harm_u), rej_harm_p)


    def _compute_benefit_from_acquiring(self, selection_threshold = None, **kwargs):
        """
        Acquiring Empirical Lift is Empirical Lift of the Marketing Product Uplift Model.
        Computes the difference of ratios of acquired&applied count to applied in deployment to ratio of acquired count in control sample to be calculated for each class or the percentiles of continuous feature.

        Parameters
        -----------
        selection_threshold : float, default = None

        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
                Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_rejected_harm : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if selection_threshold is None:
            selection_threshold = self.use_case_object.selection_threshold

        mask_list = self.feature_mask[self.curr_p_var]
        e_lift =  self.e_lift
        pred_outcome = self.pred_outcome
        
        if pred_outcome is None or e_lift is None:
            return (None, None)
        
        if 'y_pred_new' in kwargs:
            y_prob=kwargs['y_pred_new']
            e_lift = self.use_case_object._get_e_lift(y_pred_new=y_prob[1])
            pred_outcome = self.use_case_object._compute_pred_outcome(y_pred_new=y_prob)

        def _acq_benefit(pred_outcome, selection_threshold, e_lift, mask_list) :

            bools = np.array([i > selection_threshold for i in e_lift])[mask_list]
            pRcT = pred_outcome['acq_treatment'][mask_list][bools]
            pRcC = pred_outcome['acq_control'][mask_list][bools]
            benefit_acq = sum(pRcT - pRcC) / len(bools)
            return benefit_acq
        
        benefit_acq_p = _acq_benefit(pred_outcome, selection_threshold, e_lift, mask_list)
        benefit_acq_u = _acq_benefit(pred_outcome, selection_threshold, e_lift, ~mask_list)

        return ((benefit_acq_p - benefit_acq_u), benefit_acq_p)