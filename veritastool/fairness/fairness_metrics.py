from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_percentage_error, roc_auc_score, log_loss
import numpy as np
import warnings
from scipy.stats import entropy 
from ..utility import *
from ..custom import *
#from ..custom.newmetric_child import *
import concurrent.futures

class FairnessMetrics:
    """
    A class that computes all the fairness metrics

    Class Attributes
    ----------
    map_fair_metric_to_group : dictionary
        Maps the fairness metrics to its name, metric_group (classification, uplift, or regression) and type (parity or odds).
        e.g. {'equal_opportunity': ('Equal Opportunity', 'classification', 'parity'), 'equal_odds': ('Equalized Odds', 'classification', 'parity')}
    """
    map_fair_metric_to_group = {
        'disparate_impact': ('Disparate Impact', 'classification', 'ratio', True),
        'demographic_parity': ('Demographic Parity', 'classification', 'parity', True),
        'equal_opportunity': ('Equal Opportunity', 'classification', 'parity', True),
        'fpr_parity': ('False Positive Rate Parity', 'classification', 'parity', True),
        'tnr_parity': ('True Negative Rate Parity', 'classification', 'parity', True),
        'fnr_parity': ('False Negative Rate Parity', 'classification', 'parity', True),
        'ppv_parity': ('Positive Predictive Parity', 'classification', 'parity', True),
        'npv_parity': ('Negative Predictive Parity', 'classification', 'parity', True),
        'fdr_parity': ('False Discovery Rate Parity', 'classification', 'parity', True),
        'for_parity': ('False Omission Rate Parity', 'classification', 'parity', True),
        'equal_odds': ('Equalized Odds', 'classification', 'parity', True),
        'neg_equal_odds': ('Negative Equalized Odds', 'classification', 'parity', True),
        'calibration_by_group': ('Calibration by Group', 'classification', 'parity', True),
        'auc_parity': ('AUC Parity', 'classification', 'parity', False),
        'log_loss_parity': ('Log-loss Parity', 'classification', 'parity', False),
        'mi_independence': ('Mutual Information Independence', 'classification', 'information', False),
        'mi_separation': ('Mutual Information Separation', 'classification', 'information', False),
        'mi_sufficiency': ('Mutual Information Sufficiency', 'classification', 'information', False),
        'rmse_parity': ('Root Mean Squared Error Parity', 'regression', 'parity', False),
        'mape_parity': ('Mean Absolute Percentage Error Parity', 'regression', 'parity', False),
        'rejected_harm': ('Harm from Rejection', 'uplift', 'parity', True),
        'acquire_benefit': ('Benefit from Acquiring', 'uplift', 'parity', True)
    }

    for metric in newmetric.NewMetric.__subclasses__() :
        if metric.enable_flag ==True and metric.metric_type == "fair":
            map_fair_metric_to_group[metric.metric_name] =  (metric.metric_definition, metric.metric_group, metric.metric_parity_ratio,True)


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

        result : dictionary of tuples, default=None
                Data holder that stores the following for every protected variable:
                - fairness metric value, corresponding confidence interval & neutral position for all fairness metrics.
                - feature distribution

        priv_metric_val : dictionary of floats
                Contains the metric value for the privileged class for the purpose of computing the fairness threshold as 0.2 * priv_metric_val

        y_true : array of shape (n_samples,), default=None
                Ground truth target values.

        y_pred : array of shape (n_samples,), default=None
                Predicted targets as returned by classifier.

        y_prob : array of shape (n_samples,), default=None
                Predicted probabilities as returned by classifier.

        feature_mask : array of shape (n_samples,), default=None
                Array of the masked protected variable according to the privileged and unprivileged groups.

        sample_weight : array of shape (n_samples,), default=None
                Used to normalize y_true & y_pred.

        p_var : list
                List of protected variables used for fairness analysis.

        fair_metric_name: string, default  = determined by fair_concern & model_type
                Name of the primary fairness metric to be used for computations in the evaluate() and/or compile() functions

        _use_case_metrics: dictionary of lists, default="None"
                Contains all the performance & fairness metrics for credit scoring.
                {"fair ": ["fnr_parity", ...],"perf": ["balanced_accuracy, ..."]}
                Dynamically assigned during initialisation by using the _metric_group_map in Fairness/Performance Metrics class and the _model_type_to_metric above.

        """
        self.map_fair_metric_to_method = {
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
            'neg_equal_odds': self._compute_equalized_odds,
            'calibration_by_group': self._compute_calibration_by_group,
            'auc_parity': self._compute_auc_parity,
            'log_loss_parity': self._compute_log_loss_parity,
            'mi_independence': self._compute_mi_independence, 
            'mi_separation': self._compute_mi_separation, 
            'mi_sufficiency': self._compute_mi_sufficiency, 
            'rmse_parity': self._compute_rmse_parity,
            'mape_parity': self._compute_mape_parity,
            'rejected_harm': self._compute_rejected_harm,
            'acquire_benefit': self._compute_benefit_from_acquiring
        }

        #self.map_fair_metric_to_method = {'disparate_impact': self._compute_disparate_impact}
        for metric in newmetric.NewMetric.__subclasses__() : # check if we can access NewMetric.__subclasses__() only one time
            if metric.enable_flag ==True and metric.metric_type == "fair":
                self.map_fair_metric_to_method[metric.metric_name] =  metric.compute

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


    def execute_all_fair(self, n_threads):

        """
        Computes every fairness metric named inside the include_metrics list together with its associated confidence interval (dictionary), the privileged group metric value & the neutral position.

        Parameters
        ----------
        use_case_object : object
                A single initialized Fairness use case object (CreditScoring, CustomerMarketing, etc.)

        Returns
        ----------
        self.result: dictionary, default = None
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
        for i in self.p_var[0]:
            self.result[i] = {}
            idx = self.feature_mask[i]
            #unique, counts = np.unique(idx, return_counts=True)
            p_perc = sum(idx)/len(idx)
            feature_dist = { "privileged_group":p_perc, "unprivileged_group":1 - p_perc }  
            self.result[i]["feature_distribution"] = feature_dist
            self.result[i]["fair_metric_values"] = {}
            for j in self._use_case_metrics['fair']:
                if j in self.map_fair_metric_to_method.keys():
                    self.result[i]["fair_metric_values"][j] = [] 
                  

        n = len(self.use_case_object.model_params[0].y_true)
        n_threads = check_multiprocessing(n_threads)
        if n_threads >= 1:

            indices = []
            np.random.seed(123)
            for ind in range(self.use_case_object.k-1):
                indices.append(np.random.choice(n, n, replace=True))

            indexes = []
            for i in range(n_threads):
                indexes.append([])
                for x in indices[i::n_threads]:
                    indexes[i].append(x)

            threads = []

            with concurrent.futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
                #iterate through protected variables to drop one by one as part of leave-on-out
                for k in range(n_threads):
                    if n_threads == 1:
                        metric_obj = self
                    else:
                        metric_obj = deepcopy(self)
                    threads.append(executor.submit(FairnessMetrics._execute_all_fair_map, metric_obj=metric_obj, index =indexes[k]))

                for thread in threads:
                    mp_result = thread.result()
                    for i in self.p_var[0]:
                        for j in self._use_case_metrics['fair']:
                            if j in self.map_fair_metric_to_method.keys():
                                self.result[i]["fair_metric_values"][j] += mp_result[i]["fair_metric_values"][j]
            #print("multithreading done with {} threads".format(str(n_threads)))


        idx = np.array(range(n))
        FairnessMetrics._execute_all_fair_map(self, [idx])

        for i in self.p_var[0]:
            for j in self._use_case_metrics['fair']:
                if j in self.map_fair_metric_to_method.keys():
                    if self.result[i]["fair_metric_values"][j][-1][0] is None :
                        self.result[i]["fair_metric_values"][j] = (None, None, None)
                    else:
                        self.result[i]["fair_metric_values"][j] = self.result[i]['fair_metric_values'][j][-1] + (2*np.std([a_tuple[0] for a_tuple in self.result[i]["fair_metric_values"][j]]),)

    def _execute_all_fair_map(metric_obj, index):
        """
        Maps each thread's work for execute_all_fair()
        """
        for idx in index:
            metric_obj.y_true = [model.y_true[idx] for model in metric_obj.use_case_object.model_params]
            metric_obj.y_prob = [model.y_prob[idx] if model.y_prob is not None else None for model in metric_obj.use_case_object.model_params] 
            metric_obj.y_pred = [model.y_pred[idx] if model.y_pred is not None else None for model in metric_obj.use_case_object.model_params]
            metric_obj.sample_weight = [model.sample_weight[idx] if model.sample_weight is not None else None for model in metric_obj.use_case_object.model_params]
            metric_obj.e_lift = metric_obj.use_case_object.e_lift[idx] if metric_obj.use_case_object.e_lift is not None else None
            metric_obj.pred_outcome = {k: v[idx] for k, v in metric_obj.use_case_object.pred_outcome.items()} if metric_obj.use_case_object.pred_outcome is not None else {None}
            metric_obj.feature_mask = {k: v[idx] for k, v in metric_obj.use_case_object.feature_mask.items()}
                
            for i in metric_obj.p_var[0]:
                metric_obj.curr_p_var = i
                metric_obj.tp_p, metric_obj.fp_p, metric_obj.tn_p, metric_obj.fn_p, metric_obj.tp_u, metric_obj.fp_u, metric_obj.tn_u, metric_obj.fn_u  = metric_obj.use_case_object._get_confusion_matrix(metric_obj.y_true[0], metric_obj.y_pred[0], metric_obj.sample_weight[0], i, metric_obj.feature_mask)            
                for j in metric_obj._use_case_metrics['fair']:
                            metric_obj.result[i]["fair_metric_values"][j].append(metric_obj.map_fair_metric_to_method[j]())

        return metric_obj.result


    def translate_metric(self,metric_name, **kwargs ):

        """
        Computes the primary fairness metric value and its associate value for the privileged group, for the feature importance section.
        This function does not support rejection inference.

        Parameters
        ----------
        metric_name : string
                Name of fairness metric

        Other Parameters
        ----------
        kwargs : list

        Returns
        ----------
        result: dictionary, default = None
                Data holder that stores the following for every protected variable.:
                - fairness metric value, corresponding confidence interval for chosen fairness metric.
                - feature distribution

        """

        return self.map_fair_metric_to_method.get(metric_name)(**kwargs)

    def _compute_disparate_impact(self, **kwargs):
        """
        Computes the ratio of approval rate between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_disparate_impact : tuple of floats
                Fairness metric value and privileged group metric value
        """

        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u
            
        pr_p = (tp_p + fp_p) / (tp_p + fp_p + tn_p + fn_p)
        pr_u = (tp_u + fp_u) / (tp_u + fp_u + tn_u + fn_u)
        
        return (pr_p/pr_u, pr_p)


    def _compute_demographic_parity(self, **kwargs):
        """
        Computes the difference in approval rate between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_demographic_parity : tuple of floats
                Fairness metric value and privileged group metric value

        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u

        pr_p = (tp_p + fp_p) / (tp_p + fp_p + tn_p + fn_p)
        pr_u = (tp_u + fp_u) / (tp_u + fp_u + tn_u + fn_u)
        

        return (pr_p - pr_u, pr_p)



    def _compute_false_omission_rate_parity(self, **kwargs):
        """
        Computes the difference in negative predictive values between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_false_omission_rate_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u


        for_p = fn_p / (tn_p + fn_p)
        for_u = fn_u / (tn_u + fn_u)

        return (for_p - for_u, for_p)

    def _compute_false_discovery_rate_parity(self, **kwargs):
        """
        Computes the difference in false discovery rate values between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_false_discovery_rate_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u


        fdr_p = fp_p / (tp_p + fp_p)
        fdr_u = fp_u / (tp_u + fp_u)

        return (fdr_p - fdr_u, fdr_p)

    def _compute_positive_predictive_parity(self, **kwargs):
        """
        Computes the difference in positive predictive values between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_positive_predictive_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u


        ppv_p = tp_p / (tp_p + fp_p)
        ppv_u = tp_u / (tp_u + fp_u)

        return (ppv_p - ppv_u, ppv_p)

    def _compute_negative_predictive_parity(self, **kwargs):
        """
        Computes the difference in negative predictive values between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_negative_predictive_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u


        npv_p = tn_p / (tn_p + fn_p)
        npv_u = tn_u / (tn_u + fn_u)

        return (npv_p - npv_u, npv_p)

    def _compute_fnr_parity(self, **kwargs):

        """
        Computes the difference in false negative rates between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_fnr_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u

        fnr_p = fn_p / (tp_p + fn_p)
        fnr_u = fn_u / (tp_u + fn_u)

        return (fnr_p - fnr_u, fnr_p)

    def _compute_fpr_parity(self, **kwargs): ##changes to handle ci is done here, please update the other compute functions accordingly
        """
        Computes the difference in false positive rates between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_fpr_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u


        fpr_p = fp_p / (tn_p + fp_p)
        fpr_u = fp_u / (tn_u + fp_u)
        
        return (fpr_p - fpr_u, fpr_p)

    def _compute_tnr_parity(self, **kwargs):
        """
        Computes the difference in false negative rates between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_tnr_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u


        tnr_p = tn_p / (tn_p + fp_p)
        tnr_u = tn_u / (tn_u + fp_u)

        return (tnr_p - tnr_u, tnr_p)

    def _compute_equalized_odds(self, **kwargs):
        """
        Computes the equalized odds

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_equalized_odds : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u


        tpr_p = tp_p / (tp_p + fn_p)
        tpr_u = tp_u / (tp_u + fn_u)
        fpr_p = fp_p / (fp_p + tn_p)
        fpr_u = fp_u / (fp_u + tn_u)

        return (((tpr_p - tpr_u) + (fpr_p - fpr_u))/2, (tpr_p + fpr_p)/2)


    def _compute_negative_equalized_odds(self, **kwargs):
        """
        Computes the negative equalized odds

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_negative_equalized_odds : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u


        tnr_p = tn_p / (tn_p + fp_p)
        tnr_u = tn_u / (tn_u + fp_u)
        fnr_p = fn_p / (fn_p + tp_p)
        fnr_u = fn_u / (fn_u + tp_u)

        return (((tnr_p - tnr_u) + (fnr_p - fnr_u)) / 2, (tnr_p + fnr_p) / 2)

    def _compute_rmse_parity(self, **kwargs):
        """
        Computes the difference in root mean squared error between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
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

        rmse_p = mean_squared_error(np.array(y_true)[mask], np.array(y_pred)[mask], np.array(self.sample_weight[0])[mask]) ** 0.5
        rmse_u = mean_squared_error(np.array(y_true)[~mask], np.array(y_pred)[~mask], np.array(self.sample_weight[0])[~mask]) ** 0.5

        return (rmse_p - rmse_u, rmse_p)

    def _compute_mape_parity(self, **kwargs):
        """
        Computes the difference in mean average percentage error between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
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

        mape_p = mean_absolute_percentage_error(np.array(y_true)[mask], np.array(y_pred)[mask], np.array(self.sample_weight[0])[mask])
        mape_u = mean_absolute_percentage_error(np.array(y_true)[~mask], np.array(y_pred)[~mask], np.array(self.sample_weight[0])[~mask])

        return (mape_p - mape_u, mape_p)

    def _compute_wape_parity(self, **kwargs):
        """
        Computes the difference in weighted average percentage error between the privileged and unprivileged groups
    
        Other Parameters
        ----------
        y_pred_new : np.ndarray
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

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_log_loss_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        mask = self.feature_mask[self.curr_p_var]
        y_true = self.y_true[0]
        y_prob = self.y_prob[0]
        
        if y_prob is None:
            return (None, None)

        elif 'y_pred_new' in kwargs:
            y_prob=kwargs['y_pred_new'][0]

        log_loss_p = log_loss(np.array(y_true)[mask], np.array(y_prob)[mask])
        log_loss_u = log_loss(np.array(y_true)[~mask], np.array(y_prob)[~mask])

        return (log_loss_p - log_loss_u, log_loss_p)

    def _compute_auc_parity(self, **kwargs):
        """
        Computes the difference in area under roc curve between the privileged and unprivileged groups

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_auc_parity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        mask = self.feature_mask[self.curr_p_var]
        y_true = self.y_true[0]
        y_prob = self.y_prob[0]
        
        if y_prob is None:
            return (None, None)
        
            
        if 'y_pred_new' in kwargs:
            y_prob=kwargs['y_pred_new'][0]
       

        roc_auc_score_p = roc_auc_score(np.array(y_true)[mask], np.array(y_prob)[mask])
        roc_auc_score_u = roc_auc_score(np.array(y_true)[~mask], np.array(y_prob)[~mask])
        
        return (roc_auc_score_p - roc_auc_score_u, roc_auc_score_p)

    def _compute_equal_opportunity(self, **kwargs):
        """
        Computes the equal opportunity

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_equal_opportunity : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u


        tpr_p = tp_p / (tp_p + fn_p)
        tpr_u = tp_u / (tp_u + fn_u)


        return (tpr_p - tpr_u, tpr_p)

    def _compute_calibration_by_group(self, **kwargs):
        """
        Computes the calibration by group within protected variable

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_calibration_by_group : tuple of floats
                Fairness metric value and privileged group metric value
        """
        if 'y_pred_new' in kwargs:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0], curr_p_var = self.curr_p_var, feature_mask = self.feature_mask)
        else:
            tp_p, fp_p, tn_p, fn_p, tp_u, fp_u, tn_u, fn_u = self.tp_p, self.fp_p, self.tn_p, self.fn_p, self.tp_u, self.fp_u, self.tn_u, self.fn_u


        ppv_p = tp_p / (tp_p + fp_p)
        ppv_u = tp_u / (tp_u + fp_u)
        for_p = fn_p / (tn_p + fn_p)
        for_u = fn_u / (tn_u + fn_u)
        
        return (((ppv_p - ppv_u) + (for_p - for_u)) / 2, (ppv_p - ppv_u) / 2)
    
    
    def _compute_mi_independence(self, **kwargs) :
        """
        Compute Mutual Information independence

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_mi_independence : tuple of floats
                Fairness metric value and privileged group metric value

        """
        mask = self.feature_mask[self.curr_p_var]
        y_pred = self.y_pred[0]


        if 'y_pred_new' in kwargs:
            y_pred=kwargs['y_pred_new'][0]
                
        df = pd.DataFrame({'y_pred': y_pred, 'curr_p_var': mask})
        e_y_pred = self._get_entropy(df,['y_pred'])
        e_curr_p_var = self._get_entropy(df,['curr_p_var'])
        e_joint = self._get_entropy(df,['y_pred', 'curr_p_var'])
        mi_independence = e_y_pred + e_curr_p_var - e_joint
        
        return (mi_independence, None)
    
    
    def _compute_mi_separation(self, **kwargs) :
        """
        Compute Mutual Information separation

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_mi_separation : tuple of floats
                Fairness metric value and privileged group metric value
        """

        mask = self.feature_mask[self.curr_p_var]
        y_true = self.y_true[0]
        y_pred = self.y_pred[0]

        if 'y_pred_new' in kwargs:
            y_pred=kwargs['y_pred_new'][0]     
        
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'curr_p_var': mask})
        e_y_true_curr_p_var = self._get_entropy(df,['y_true', 'curr_p_var'])
        e_y_pred_curr_p_var = self._get_entropy(df,['y_pred', 'curr_p_var'])
        e_y_true_y_pred_curr_p_var = self._get_entropy(df,['y_true', 'y_pred', 'curr_p_var'])
        e_y_true = self._get_entropy(df,['y_true'])
        mi_separation = e_y_true_curr_p_var + e_y_pred_curr_p_var - e_y_true_y_pred_curr_p_var - e_y_true
            
        return (mi_separation, None)
    

    def _compute_mi_sufficiency(self, **kwargs) :
        """
        Compute Mutual Information sufficiency

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_mi_sufficiency : tuple of floats
                Fairness metric value and privileged group metric value
        """
        mask = self.feature_mask[self.curr_p_var]
        y_true = self.y_true[0]
        y_pred = self.y_pred[0]

        if 'y_pred_new' in kwargs:
            y_pred=kwargs['y_pred_new'][0]     
        
        df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'curr_p_var': mask})
        e_y_pred_curr_p_var = self._get_entropy(df,['y_pred', 'curr_p_var'])
        e_y_true_y_pred = self._get_entropy(df,['y_true', 'y_pred'])
        e_y_true_y_pred_curr_p_var = self._get_entropy(df,['y_true', 'y_pred', 'curr_p_var'])
        e_y_pred = self._get_entropy(df,['y_pred'])
        mi_sufficiency = e_y_pred_curr_p_var + e_y_true_y_pred - e_y_true_y_pred_curr_p_var - e_y_pred
            
        return (mi_sufficiency, None)
    

    def _get_entropy(self, df, columns) :
        """
        Compute the entropy

        Parameters
        -----------
        df : pandas DataFrame
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
        y_pred_new : np.ndarray
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
        y_pred_new : np.ndarray
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

