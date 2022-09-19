from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_percentage_error, roc_auc_score, log_loss, brier_score_loss, balanced_accuracy_score, f1_score
from sklearn.calibration import calibration_curve
from ..util.utility import *
import numpy as np
from numpy import ma
from .newmetric import *
from ..config.constants import Constants
import concurrent.futures

class PerformanceMetrics:
    """
    A class that computes all the performance metrics

    Class Attributes
    ------------------------
    map_perf_metric_to_group : dict
            Maps the performance metric names to their corresponding full names, metric group eg classification or regression metric types, and whether it can be a primary metric
    """
    map_perf_metric_to_group = {
                                'selection_rate':('Selection Rate', 'classification', True),
                                'accuracy': ('Accuracy', 'classification', True),
                                'balanced_acc': ('Balanced Accuracy', 'classification', True),
                                'recall': ('Recall', 'classification', True),
                                'precision': ('Precision', 'classification', True),
                                'f1_score': ('F1 Score', 'classification', True),
                                'tnr': ('True Negative Rate', 'classification', True),
                                'fnr': ('False Negative Rate', 'classification', True),
                                'npv': ('Negative Predictive Value', 'classification', True),
                                'roc_auc': ('ROC AUC Score', 'classification', True),
                                'log_loss': ('Log-loss', 'classification', True),
                                'rmse': ('Root Mean Squared Error', 'regression', True),
                                'mape': ('Mean Absolute Percentage Error', 'regression', True),
                                'wape': ('Weighted Absolute Percentage Error', 'regression', True),
                                'emp_lift': ('Empirical Lift', 'uplift', True),
                                'expected_profit': ('Expected Profit Lift', 'uplift', True),
                                'expected_selection_rate':('Expected Selection Rate', 'uplift', True)
                                }
   
    def __init__(self, use_case_object):
        """
        Parameters
        ------------------------
        use_case_object : object
                Object is initialised in use case classes.

        Instance Attributes
        ------------------------
        map_perf_metric_to_method : dict
                Maps the performance metric names to their corresponding functions.
        
        result : dict of dict, default=None
                Stores the following:
                - percentage distribution of classes (dictionary)
                - every performance metric named inside the include_metrics list together with its associated confidence interval (dictionary)
                - calibration curve (dictionary)
                - performance dynamic values (dictionary)

        y_true : numpy.ndarray, default=None
                Ground truth target values.

        y_pred : numpy.ndarray, default=None
                Predicted targets as returned by classifier.

        y_train :numpy.ndarray, default=None
                Ground truth for training data.

        y_prob : numpy.ndarray, default=None
                Predicted probabilities as returned by classifier. 
                For uplift models, L = 4. Else, L = 1 where shape is (n_samples, L)

        sample_weight : numpy.ndarray, default=None
                Used to normalize y_true & y_pred.

        perf_metric_name : str, default=None
                Performance metric name

        _use_case_metrics : dict of lists, default=None
                Contains all the performance & fairness metrics for a use case.
                Dynamically assigned during initialisation by using the _metric_group_map in Fairness/Performance Metrics class and the _model_type_to_metric.

        revenue : float, default=None
                Revenue gained per customer

        treatment_cost : float, default=None
                Cost of the marketing treatment per customer

        label_size : int, default=None
                Number of labels allowed
        """
        self.result = None

        self.map_perf_metric_to_method = {'rmse': self._compute_rmse,
                                          'mape': self._compute_mape,
                                          'wape': self._compute_wape, 
                                          'emp_lift': self._compute_emp_lift,
                                          'expected_profit': self._compute_expected_profit,
                                          'expected_selection_rate': self._compute_expected_selection_rate}
        self.map_perf_metric_to_method_optimized = {'selection_rate': self._compute_selection_rate,
                                          'accuracy': self._compute_accuracy,
                                          'balanced_acc': self._compute_balanced_accuracy,
                                          'recall': self._compute_recall,
                                          'precision': self._compute_precision,
                                          'f1_score': self._compute_f1_score,
                                          'tnr': self._compute_tnr,
                                          'fnr': self._compute_fnr,
                                          'npv': self._compute_negative_predictive_value,
                                          'roc_auc': self._compute_roc_auc_score,
                                          'log_loss': self._compute_log_loss,
                                          }

        for metric in NewMetric.__subclasses__() :
            if metric.enable_flag ==True and metric.metric_type == "perf":
                self.map_perf_metric_to_method[metric.metric_name] =  metric.compute
                self.map_perf_metric_to_group[metric.metric_name] = (metric.metric_definition, metric.metric_group, False)
                if metric.metric_name not in use_case_object._use_case_metrics["perf"]:
                    use_case_object._use_case_metrics["perf"].append(metric.metric_name)
                    
        self.y_true = None
        self.y_prob = None
        self.y_pred = None
        self.y_train = None
        self.sample_weight = None
        self.perf_metric_name = None
        self._use_case_metrics = None
        self.treatment_cost = None
        self.revenue = None
        self.label_size = None
        self.use_case_object = use_case_object

    def execute_all_perf(self, n_threads, seed, eval_pbar):
        """
        Computes the following:
                - every performance metric named inside the include_metrics list together with its associated confidence interval (dictionary)
                - calibration brier loss score (float)
                - percentage distribution of classes (dictionary)
                - performance dynamic values (dictionary)
                - weighted confusion matrix (dictionary)

        Parameters
        ----------
        n_threads : int
                Number of currently active threads of a job
                
        seed : int
                Used to initialize the random number generator.

        eval_pbar : tqdm object
                Progress bar

        Returns
        ----------
        self.result : dict of lists
                Stores the class distribution, weighted confusion matrix, performance metric values and performance dynamics results
        """

        self.perf_metric_name = self.use_case_object.perf_metric_name
        self._use_case_metrics = self.use_case_object._use_case_metrics
        self.y_train = [model.y_train for model in self.use_case_object.model_params]

        #initialize result structure
        self.result = {}
        self.result["perf_metric_values"] = {}
        for j in self._use_case_metrics['perf']:
                if j in list(self.map_perf_metric_to_method.keys()) + list(self.map_perf_metric_to_method_optimized.keys()):
                    self.result["perf_metric_values"][j] = []
        #update progress bar by 10
        eval_pbar.update(10)

        n_threads = check_multiprocessing(n_threads)
        n = len(self.use_case_object.model_params[0].y_true)
        #split k into k-1 times of random indexing compute and 1 time of original array compute
        if n_threads >= 1 and self.use_case_object.k > 1:
            indices = []
            np.random.seed(seed)
            for ind in range(self.use_case_object.k-1):
                indices.append(np.random.choice(n, n, replace=True))

            threads = []
            indexes = []
            for i in range(n_threads):
                indexes.append([])
                for x in indices[i::n_threads]:
                    indexes[i].append(x)

            worker_progress = 24/n_threads
            with concurrent.futures.ThreadPoolExecutor(max_workers = n_threads) as executor:
                #iterate through protected variables to drop one by one as part of leave-on-out
                for k in range(n_threads):
                    if n_threads == 1:
                        metric_obj = self
                    else:
                        metric_obj = deepcopy(self)
                    #submit each thread's work to thread pool
                    if len(indexes[k]) > 0:
                        threads.append(executor.submit(PerformanceMetrics._execute_all_perf_map, metric_obj=metric_obj, index=indexes[k], eval_pbar=eval_pbar, worker_progress=worker_progress))

                #retrive results from each thread
                for thread in threads:
                    for key, v in thread.result().items():
                        self.result['perf_metric_values'][key] = self.result['perf_metric_values'][key] + v
        else:
            #if multithreading is not triggered, directly update the progress bar by 24
            eval_pbar.update(24)

        #run 1 time of original array to compute performance metrics
        PerformanceMetrics._execute_all_perf_map(self, [np.arange(n)], eval_pbar, 1)

        #generate the final performace metrics values and their CI based on k times of computation
        for j in self.result['perf_metric_values'].keys():
            if self.result['perf_metric_values'][j][-1] is None :
                self.result['perf_metric_values'][j] = (None, None)
            else :
               self.result['perf_metric_values'][j] = (self.result['perf_metric_values'][j][-1],  2*np.std(self.result['perf_metric_values'][j]))
        self.label_size = self.use_case_object._model_type_to_metric_lookup[self.use_case_object.model_params[0].model_type][1]
        self.result["class_distribution"] = self._get_class_distribution(self.y_true[-1], self.use_case_object.model_params[-1].pos_label2)
        self.result['weighted_confusion_matrix'] = { "tp":self.tp_s[0][0], "fp": self.fp_s[0][0], "tn": self.tn_s[0][0], "fn": self.fn_s[0][0] }
        self.result["calibration_curve"] = self._calibration_func(self.y_true[0], self.y_prob[0], n_bins=10)
        self.result["perf_dynamic"] = self._performance_dynamics()
        eval_pbar.update(6)

    def _execute_all_perf_map(metric_obj, index, eval_pbar, worker_progress):
        """
        Maps each thread's work for execute_all_perf()
        Parameters
        ----------
        metric_obj : PerformanceMetrics object
        index : list
        eval_pbar : tqdm object
                Progress bar
        worker_progress : int
                Progress bar progress for each thread
        """

        #get each iteration's progress in 2 decimals to update the progress bar
        prog = round(worker_progress/(len(index)),2)

        #list to store all np arrays and combine for vectorization
        metric_obj.y_trues = []
        metric_obj.y_probs = []
        metric_obj.y_preds = []
        metric_obj.sample_weights = []

        for idx in index:
            metric_obj.y_true = [model.y_true[idx] for model in metric_obj.use_case_object.model_params]
            metric_obj.y_prob = [model.y_prob[idx] if model.y_prob is not None else None for model in metric_obj.use_case_object.model_params] 
            metric_obj.y_pred = [model.y_pred[idx] if model.y_pred is not None else None for model in metric_obj.use_case_object.model_params]
            metric_obj.sample_weight = [model.sample_weight[idx] if model.sample_weight is not None else None for model in metric_obj.use_case_object.model_params]
            metric_obj.e_lift = metric_obj.use_case_object.e_lift[idx] if metric_obj.use_case_object.e_lift is not None else None
            metric_obj.pred_outcome = {k: v[idx] for k, v in metric_obj.use_case_object.pred_outcome.items()} if metric_obj.use_case_object.pred_outcome is not None else {None}

            metric_obj.y_trues.append(metric_obj.y_true)
            metric_obj.y_probs.append(metric_obj.y_prob)
            metric_obj.y_preds.append(metric_obj.y_pred)
            metric_obj.sample_weights.append(metric_obj.sample_weight)

            for j in metric_obj._use_case_metrics['perf']:
                if j in metric_obj.map_perf_metric_to_method.keys():
                    metric_obj.result['perf_metric_values'][j].append(metric_obj.map_perf_metric_to_method[j](obj=metric_obj))

        metric_obj.y_trues = np.array(metric_obj.y_trues)
        metric_obj.y_probs = np.array(metric_obj.y_probs)
        metric_obj.y_preds = np.array(metric_obj.y_preds)
        metric_obj.sample_weights = np.array(metric_obj.sample_weights)
        
        metric_obj.tp_s, metric_obj.fp_s, metric_obj.tn_s, metric_obj.fn_s = \
            metric_obj.use_case_object._get_confusion_matrix_optimized(
                    metric_obj.y_trues,
                    metric_obj.y_preds, 
                    metric_obj.sample_weights
                ) 

        for j in metric_obj._use_case_metrics['perf']:
            if j in metric_obj.map_perf_metric_to_method_optimized.keys():
                metric_obj.result["perf_metric_values"][j] += metric_obj.map_perf_metric_to_method_optimized[j](obj=metric_obj)

        return metric_obj.result['perf_metric_values']

    def translate_metric(self, metric_name, **kwargs):
        """
        Computes the primary performance metric value with its confidence interval for the feature importance section. 
        This function does not support rejection inference.

        Parameters
        ----------
        metric_name : str
            Name of fairness metric

        Other parameters
        -----------
        kwargs : list

        Returns
        ----------
        perf_metric_values : dict of tuples
            Stores both the performance metric value and the corresponding confidence interval for every metric in include_metrics
        """
        if metric_name in self.map_perf_metric_to_method.keys():
            return self.map_perf_metric_to_method[metric_name](**kwargs)
        if metric_name in self.map_perf_metric_to_method_optimized.keys():
            return self.map_perf_metric_to_method_optimized[metric_name](**kwargs)
        
    def _compute_selection_rate(self, **kwargs):
        """
        Computes the selection_rate value

        Returns
        ----------
        _compute_accuracy : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1)
            )
            selection_rate = (tp + fp) / (tp + tn + fp + fn)
            return selection_rate[0][0]
        else:
            selection_rate = (self.tp_s + self.fp_s) / (self.tp_s + self.tn_s + self.fp_s + self.fn_s)
            return selection_rate.reshape(-1).tolist()

    def _compute_accuracy(self, **kwargs):
        """
        Computes the accuracy value

        Returns
        ----------
        _compute_accuracy : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1)
            )
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            return accuracy[0][0]
        else:
            accuracy = (self.tp_s + self.tn_s) / (self.tp_s + self.tn_s + self.fp_s + self.fn_s)
            return accuracy.reshape(-1).tolist()

    def _compute_balanced_accuracy(self, **kwargs):
        """
        Computes balanced accuracy score

        Returns
        ----------
        _compute_balanced_accuracy : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1)
            )
            balanced_accuracy = ((tp/(tp+fn)) + (tn/(tn+fp)))/2 
            return balanced_accuracy[0][0]
        else:
            balanced_accuracy = ((self.tp_s/(self.tp_s+self.fn_s)) + (self.tn_s/(self.tn_s+self.fp_s)))/2  
            return balanced_accuracy.reshape(-1).tolist()

    def _compute_f1_score(self, **kwargs):
        """
        Computes F1 score

        Returns
        ----------
        _compute_f1_score : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1)
            )
            f1_scr = 2 * ((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + tp / (tp + fn))
            return f1_scr[0][0]
        else:
            f1_scr = 2 * ((self.tp_s / (self.tp_s + self.fp_s)) * (self.tp_s / (self.tp_s + self.fn_s))) / \
                     ((self.tp_s / (self.tp_s + self.fp_s)) + self.tp_s / (self.tp_s + self.fn_s)) 
            return f1_scr.reshape(-1).tolist()

    def _compute_precision(self, **kwargs):
        """
        Computes the precision

        Returns
        ----------
        _compute_precision : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1)
            )
            precision = tp / (tp + fp)
            return precision[0][0]
        else:
            precision = self.tp_s / (self.tp_s + self.fp_s)
            return precision.reshape(-1).tolist()

    def _compute_recall(self, **kwargs):
        """
        Computes the recall

        Returns
        ----------
        _compute_recall : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1)
            )
            recall = tp / (tp + fn)
            return recall[0][0]
        else:
            recall = self.tp_s / (self.tp_s + self.fn_s)
            return recall.reshape(-1).tolist()

    def _compute_tnr(self, **kwargs):
        """
        Computes the true negative rate or specificity

        Returns
        ----------
        _compute_tnr : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1)
            )
            tnr = tn / (tn + fp)
            return tnr[0][0]
        else:
            tnr = self.tn_s / (self.tn_s + self.fp_s)
            return tnr.reshape(-1).tolist()

    def _compute_fnr(self, **kwargs):
        """
        Computes the false negative rate or miss-rate
  
        Returns
        ----------
        _compute_fnr : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1)
            )
            fnr = fn / (tp + fn)
            return fnr[0][0]
        else:
            fnr = self.fn_s / (self.tp_s + self.fn_s)
            return fnr.reshape(-1).tolist()

    def _compute_emp_lift(self, selection_threshold = None, **kwargs):
        """
        Computes empirical lift between treatment and control group

        Parameters
        ----------
        selection_threshold : float, default = None

        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_emp_lift : float
                The performance metric value
        """
        y_true = self.y_true[1]
        e_lift = self.e_lift
        
        if 'y_pred_new' in kwargs:
            y_prob =kwargs['y_pred_new']
            e_lift = self.use_case_object._get_e_lift(y_pred_new=y_prob[1])
            
        if e_lift is None:
            return (None, None)
        
        if selection_threshold is None:
            selection_threshold = self.use_case_object.selection_threshold
            
        bools = [i > selection_threshold for i in e_lift]
        Ntr = sum(y_true[bools] == "TR")
        Ntn = sum(y_true[bools] == "TN")
        pRcT = Ntr / (Ntr + Ntn)
        Ncr = sum(y_true[bools] == "CR")
        Ncn = sum(y_true[bools] == "CN")
        pRcC = Ncr / (Ncr + Ncn)
        emp_lift = pRcT - pRcC

        return emp_lift

    def _compute_expected_profit(self, selection_threshold = None, **kwargs):
        """
        Computes expected profit from the revenue and cost values

        Parameters
        ----------
        selection_threshold : float, default=None

        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_expected_profit : float
                The performance metric value
        """
        if self.use_case_object.spl_params['revenue'] is None or self.use_case_object.spl_params['treatment_cost'] is None :
            return None
        
        e_lift = self.e_lift
        pred_outcome = self.pred_outcome
    
        if 'y_pred_new' in kwargs:
            y_prob =kwargs['y_pred_new']
            e_lift = self.use_case_object._get_e_lift(y_pred_new=y_prob[1])
            pred_outcome = self.use_case_object._compute_pred_outcome(y_pred_new=y_prob)
            
        if pred_outcome is None or e_lift is None:
            return None
        
        if selection_threshold is None:
            selection_threshold = self.use_case_object.selection_threshold
            
        bools = [i > selection_threshold for i in e_lift]
        pRcT = pred_outcome['acq_treatment'][bools]
        pRcC = pred_outcome['acq_control'][bools]
        profit_RcT = pRcT * self.use_case_object.spl_params['revenue']- self.use_case_object.spl_params['treatment_cost']
        profit_RcC = pRcC * self.use_case_object.spl_params['revenue']
        profit = sum(profit_RcT - profit_RcC)

        return profit

    def _compute_expected_selection_rate(self, selection_threshold = None, **kwargs):
        """
        Computes expected selection rate
    
        Parameters
        ----------
        selection_threshold : float, default=None

        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_expected_selection_rate : float
            The performance metric value
        """
        e_lift = self.e_lift
        
        if 'y_pred_new' in kwargs:
            y_prob = kwargs['y_pred_new']
            e_lift = self.use_case_object._get_e_lift(y_pred_new=y_prob[1])
            
        if self.e_lift is None:
            return None
        
        if selection_threshold is None:
            selection_threshold = self.use_case_object.selection_threshold
            
        bools = [i > selection_threshold for i in e_lift]
        bools_avg = sum(bools)/len(bools)
        return bools_avg

    def _compute_negative_predictive_value(self, **kwargs):
        """
        Computes the negative predictive value

        Returns
        ----------
        _compute_negative_predictive_value : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix_optimized(
                y_true=np.array(self.y_true[0]).reshape(1, 1, -1), 
                y_pred=np.array(kwargs['y_pred_new'][0]).reshape(1, 1, -1),
                sample_weight=np.array(self.sample_weight[0]).reshape(1, 1, -1)
            )
            npv = tn / (tn + fn)
            return npv[0][0]
        else:
            npv = self.tn_s / (self.tn_s + self.fn_s)
            return npv.reshape(-1).tolist()

    def _compute_rmse(self, **kwargs):
        """
        Computes root mean squared error

        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_rmse : float
                The performance metric value
        """
        y_true = self.y_true[0]
        y_pred = self.y_pred[0]
      
        if 'y_pred_new' in kwargs:
            y_pred=kwargs['y_pred_new'][0]
      
        rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, sample_weight=self.sample_weight[0]) ** 0.5

        return rmse

    def _compute_mape(self, **kwargs):
        """
        Computes the mean absolute percentage error

        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_mape : float
                The performance metric value
        """
        y_true = self.y_true[0]
        y_pred = self.y_pred[0]
       
        if 'y_pred_new' in kwargs:
            y_pred=kwargs['y_pred_new'][0]

        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred, sample_weight=self.sample_weight[0])
        return mape

    def _compute_wape(self, **kwargs):
        """
        Computes the weighted average percentage error
    
        Other Parameters
        ----------
        y_pred_new : numpy.ndarray
            Copy of predicted targets as returned by classifier.
    
        Returns
        ----------
        _compute_wape : float
                The performance metric value
        """
        y_true = self.y_true[0]
        y_pred = self.y_pred[0]
    
        if 'y_pred_new' in kwargs:
            y_pred = kwargs['y_pred_new'][0]
    
        wape = np.sum(np.absolute(np.subtract(y_true, y_pred)))/ np.sum(y_true)

        return wape

    def _compute_roc_auc_score(self, **kwargs):
        """
        Computes the ROC score

        Returns
        ----------
        _compute_roc_auc_score : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            y_true = self.y_true[0]
            y_prob = kwargs['y_pred_new'][0]
            if self.sample_weight[0] is None :
                roc_auc = roc_auc_score(y_true=y_true, y_score=y_prob)
            else: 
                sample_weight = self.sample_weight[0]
                roc_auc = roc_auc_score(y_true=y_true, y_score=y_prob, sample_weight=self.sample_weight[0])
            return roc_auc
        else:
            idx = self.y_probs.argsort(axis=2)[:,:,::-1] # sort by descending order
            y_probs = np.take_along_axis(self.y_probs, idx, axis=2)
            y_trues = np.take_along_axis(self.y_trues, idx, axis=2)
            TPR = np.cumsum(y_trues, axis=2)/np.sum(y_trues, axis=2, keepdims=True)
            FPR = np.cumsum(1-y_trues, axis=2)/np.sum(1-y_trues, axis=2, keepdims=True)
            TPR = np.append(np.zeros((TPR.shape[0],TPR.shape[1],1)), TPR, axis=2) # append starting point (0)
            FPR = np.append(np.zeros((FPR.shape[0],FPR.shape[1],1)), FPR, axis=2)
            roc_auc = np.trapz(TPR, FPR, axis=2)
            return roc_auc.reshape(-1).tolist()

    def _compute_log_loss(self, **kwargs):
        """
        Computes the log loss score

        Returns
        ----------
        _compute_log_loss : list of float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            y_true = self.y_true[0]
            y_prob = kwargs['y_pred_new'][0]
            if self.sample_weight[0] is None :
                log_loss_score = log_loss(y_true=y_true, y_pred=y_prob)
            else: 
                sample_weight = self.sample_weight[0]
                log_loss_score = log_loss(y_true=y_true, y_pred=y_prob, sample_weight=self.sample_weight[0])
            return log_loss_score
        else:
            log_loss_score = -(self.y_trues*ma.log(self.y_probs) + (1-self.y_trues)*ma.log(1-self.y_probs))
            log_loss_score = np.sum(log_loss_score, 2)/log_loss_score.shape[2]
            return log_loss_score.reshape(-1).tolist()

    def _performance_dynamics(self):
        """
        Computes the dynamic performance metrics based on different threshold values

        Returns
        ----------
        d : dict
                100 values of selection rate, balanced accuracy, F1 score, expected_profit and threshold
        """
        metric_group = self.map_perf_metric_to_group.get(self.perf_metric_name)[1]
        if self.y_prob[0] is None or metric_group == 'regression':
            return None
        else:
            d = {}
            d['perf_metric_name'] = []
            d['threshold'] = []
            d['perf'] = []
            d['selection_rate'] = []
            if metric_group == 'classification':
                threshold = np.linspace(Constants().classify_min_threshold, Constants().classify_max_threshold, Constants().perf_dynamics_array_size)
                d['threshold'] = threshold
                
                #TODO: Optimize, keep fewer objects in memory
                asc_score_indices = np.argsort(self.y_prob[0])
                desc_score_indices = asc_score_indices[::-1]
                desc_sorted_score = self.y_prob[0][desc_score_indices]
                desc_sorted_true = self.y_true[0][desc_score_indices]
                asc_sorted_score = self.y_prob[0][asc_score_indices]
                asc_sorted_true = self.y_true[0][asc_score_indices]

                desc_search_idx = np.searchsorted(-desc_sorted_score, -threshold).astype(int)
                asc_search_idx = np.searchsorted(asc_sorted_score, threshold)

                true_positives = np.cumsum(desc_sorted_true)[desc_search_idx-1]
                false_positives = desc_search_idx - true_positives
                true_negatives = asc_search_idx - np.cumsum(asc_sorted_true)[asc_search_idx-1]
                false_negatives = asc_search_idx - true_negatives

                d['perf_metric_name'] = 'balanced_acc'
                d["selection_rate"] = ((true_positives + false_positives) / (true_positives + false_positives + true_negatives + false_negatives)).tolist()
                d["perf"] = np.mean([true_positives / (true_positives+false_negatives), true_negatives / (true_negatives + false_positives)], axis=0).tolist()
            elif metric_group == 'uplift':
                if self.y_prob[1] is None:
                    return None
                else:
                    threshold = np.linspace(Constants().uplift_min_threshold, Constants().uplift_max_threshold, Constants().perf_dynamics_array_size)
                    if self.perf_metric_name == 'emp_lift':
                        d['perf_metric_name'] = 'emp_lift'
                        d['threshold'] = threshold
                        for i in range(len(threshold)):
                            d['selection_rate'] += [self._compute_expected_selection_rate(threshold[i])]
                            d['perf'] += [self._compute_emp_lift(threshold[i])]
                    else:   
                        d['perf_metric_name'] = 'expected_profit'
                        d['threshold'] = threshold
                        for i in range(len(threshold)):
                            d['selection_rate'] += [self._compute_expected_selection_rate(threshold[i])]
                            d['perf'] += [self._compute_expected_profit(threshold[i])]
            else:
                return None

        return d

    def _calibration_func(self, y_true, y_prob, n_bins=10):
        """
        Calculates the points for calibration curve over a bin of values
        and the calibration score based on brier loss score.
        Returns results in the calibration_curve_bin dictionary.

        Parameters
        ----------
        y_true: numpy.ndarray
            Ground truth target values.

        y_prob : numpy.ndarray, default=None
                Predicted probabilities as returned by classifier. 
                For uplift models, L = 4. Else, L = 1 where shape is (n_samples, L)

        n_bins : int, default=10
            Number of equal-width bins in the range

        Returns
        ----------
        calibration_curve_bin : dict
            Contains prob_true, prob_pred and score as floats
        """
        metric_group = self.map_perf_metric_to_group.get(self.perf_metric_name)[1]
        if y_prob is None or self.label_size > 2 or metric_group == 'regression':
            calibration_curve_bin = None
        else:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins = n_bins)
            score = brier_score_loss(y_true=y_true, y_prob=y_prob)

            calibration_curve_bin = {"prob_true": prob_true, "prob_pred": prob_pred, "score": score}

        return calibration_curve_bin

    def _get_class_distribution(self, y_true, pos_label = 1):
        """
        Calculates the proportion of favourable and unfavourable labels in y_true.
        Parameters
        ----------
        y_true: numpy.ndarray
            Ground truth target values.

        pos_label : list, default=1
            Label values which are considered favorable.
            For all model types except uplift, converts the favourable labels to 1 and others to 0.
            For uplift, user is to provide 2 label names e.g. [["a"], ["b"]] in fav label. The first will be mapped to treatment responded (TR) & second to control responded (CR).

        Returns
        ----------
        y_true_counts : dict
            Dictionary of proportion of classes
        """
        if self.label_size == -1:
            return None
        else:
            y_true_counts = pd.Series(y_true).value_counts(normalize = True)
            y_true_counts = y_true_counts.reset_index().replace({1: 'pos_label', 0:'neg_label'}).set_index('index')
            return y_true_counts[0].to_dict()