from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_percentage_error, roc_auc_score, log_loss, brier_score_loss, balanced_accuracy_score, f1_score
from sklearn.calibration import calibration_curve
from ..util.utility import *
import numpy as np
from .newmetric import *
from ..config.constants import Constants
import concurrent.futures

class PerformanceMetrics:
    """
    A class that computes all the performance metrics

    Class Attributes
    ------------------------
    map_perf_metric_to_group : dictionary
            Maps the performance metric names to their corresponding full names, metric group eg classification or regression metric types, and whether it can be a primary metric
    """
    map_perf_metric_to_group = {'selection_rate':('Selection Rate', 'classification', True),
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
        map_perf_metric_to_method : dictionary
                Maps the performance metric names to their corresponding functions.
        
        result : dictionary of tuples, default=None
                Stores the following:
                - percentage distribution of classes (dictionary)
                - every performance metric named inside the include_metrics list together with its associated confidence interval (dictionary)
                - calibration curve (dictionary)
                - performance dynamic values (dictionary)

        y_true : array of shape (n_samples,), default=None
                Ground truth target values.

        y_pred : array of shape (n_samples,), default=None
                Predicted targets as returned by classifier.

        y_train : array of shape (n_samples,), default=None
                Ground truth for training data.

        y_prob : array of shape (n_samples, L), default=None
                Predicted probabilities as returned by classifier. For uplift models, L = 4. Else, L = 1.

        sample_weight : array of shape (n_samples,), default=None
                Used to normalize y_true & y_pred.

        perf_metric_name : string, default=None
                Performance metric name

        _use_case_metrics : dictionary of lists, default=None
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

        self.map_perf_metric_to_method = {'selection_rate': self._compute_selection_rate,
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
                                          'rmse': self._compute_rmse,
                                          'mape': self._compute_mape,
                                          'wape': self._compute_wape, 
                                          'emp_lift': self._compute_emp_lift,
                                          'expected_profit': self._compute_expected_profit,
                                          'expected_selection_rate': self._compute_expected_selection_rate}

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
        self.result : dictionary of lists
                Stores the class distribution, weighted confusion matrix, performance metric values and performance dynamics results
        """

        self.perf_metric_name = self.use_case_object.perf_metric_name
        self._use_case_metrics = self.use_case_object._use_case_metrics
        self.y_train = [model.y_train for model in self.use_case_object.model_params]

        #initialize result structure
        self.result = {}
        self.result["perf_metric_values"] = {}
        for j in self._use_case_metrics['perf']:
                if j in self.map_perf_metric_to_method.keys():
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
        self.result['weighted_confusion_matrix'] = { "tp":self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn }
        self.result["calibration_curve"] = self._calibration_func(self.y_true[0], self.y_prob[0], n_bins=10)
        self.result["perf_dynamic"] = self._performance_dynamics()
        eval_pbar.update(6)

    def _execute_all_perf_map(metric_obj, index, eval_pbar, worker_progress):
        """
        Maps each thread's work for execute_all_perf()
        Parameters
        ----------
        metric_obj : PerformanceMetrics object
        index : array of shape (n,m)
        eval_pbar : tqdm object
                Progress bar
        worker_progress : int
                Progress bar progress for each thread
        """
        #get each iteration's progress in 2 decimals to update the progress bar
        prog = round(worker_progress/(len(index)),2)
        for idx in index:
            #prepare data
            metric_obj.y_true = [model.y_true[idx] for model in metric_obj.use_case_object.model_params]
            metric_obj.y_prob = [model.y_prob[idx] if model.y_prob is not None else None for model in metric_obj.use_case_object.model_params] 
            metric_obj.y_pred = [model.y_pred[idx] if model.y_pred is not None else None for model in metric_obj.use_case_object.model_params]
            metric_obj.sample_weight = [model.sample_weight[idx] if model.sample_weight is not None else None for model in metric_obj.use_case_object.model_params]
            metric_obj.e_lift = metric_obj.use_case_object.e_lift[idx] if metric_obj.use_case_object.e_lift is not None else None
            metric_obj.pred_outcome = {k: v[idx] for k, v in metric_obj.use_case_object.pred_outcome.items()} if metric_obj.use_case_object.pred_outcome is not None else {None}

            #compute performace metrics
            metric_obj.tp, metric_obj.fp, metric_obj.tn, metric_obj.fn = metric_obj.use_case_object._get_confusion_matrix(metric_obj.y_true[0], metric_obj.y_pred[0], metric_obj.sample_weight[0])            
            for j in metric_obj._use_case_metrics['perf']:
                if j in metric_obj.map_perf_metric_to_method.keys():
                    metric_obj.result['perf_metric_values'][j].append(metric_obj.map_perf_metric_to_method[j](obj=metric_obj))
            eval_pbar.update(prog)
        return metric_obj.result['perf_metric_values']

    def translate_metric(self, metric_name, **kwargs):
        """
        Computes the primary performance metric value with its confidence interval for the feature importance section. 
        This function does not support rejection inference.

        Parameters
        ----------
        metric_name : string
            Name of fairness metric

        Other parameters
        -----------
        kwargs : list

        Returns
        ----------
        perf_metric_values : dictionary of tuples
            Stores both the performance metric value and the corresponding confidence interval for every metric in include_metrics
        """
        return self.map_perf_metric_to_method[metric_name](**kwargs)
    
    def _compute_selection_rate(self, **kwargs):
        """
        Computes the selection_rate value

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_accuracy : float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0])
        else:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
            
        selection_rate = (tp + fp) / (tp + tn + fp + fn)

        return selection_rate

    def _compute_accuracy(self, **kwargs):
        """
        Computes the accuracy value

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_accuracy : float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0])
        else:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
            
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return accuracy

    def _compute_balanced_accuracy(self, **kwargs):
        """
        Computes balanced accuracy score

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_balanced_accuracy : float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0])
        else:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

        balanced_accuracy = ((tp/(tp+fn)) + (tn/(tn+fp)))/2        
        
        return balanced_accuracy

    def _compute_f1_score(self, **kwargs):
        """
        Computes F1 score

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_f1_score : float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0])
        else:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
            
        f1_scr = 2 * ((tp / (tp + fp)) * (tp / (tp + fn))) / ((tp / (tp + fp)) + tp / (tp + fn))
        return f1_scr

    def _compute_precision(self, **kwargs):
        """
        Computes the precision

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_precision : float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0])
        else:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
        
        precision = tp / (tp + fp)
        return precision

    def _compute_recall(self, **kwargs):
        """
        Computes the recall

        Other Parameters
        ----------
         y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_recall : float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0])
        else:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn

        recall = tp / (tp + fn)
        return recall

    def _compute_tnr(self, **kwargs):
        """
        Computes the true negative rate or specificity

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_tnr : float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0])
        else:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
            
        tnr = tn / (tn + fp)
        return tnr

    def _compute_fnr(self, **kwargs):
        """
        Computes the false negative rate or miss-rate
        
        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_fnr : float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0])
        else:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
            
        fnr = fn / (tp + fn)
        return fnr

    def _compute_emp_lift(self, selection_threshold = None, **kwargs):
        """
        Computes empirical lift between treatment and control group

        Parameters
        ----------
        selection_threshold : float, default = None

        Other Parameters
        ----------
        y_pred_new : np.ndarray
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
        y_pred_new : np.ndarray
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
        y_pred_new : np.ndarray
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

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_negative_predictive_value : float
                The performance metric value
        """
        if 'y_pred_new' in kwargs:
            tp, fp, tn, fn = self.use_case_object._get_confusion_matrix(y_true=self.y_true[0], y_pred=kwargs['y_pred_new'][0], sample_weight = self.sample_weight[0])
        else:
            tp, fp, tn, fn = self.tp, self.fp, self.tn, self.fn
            
        npv = tn / (tn + fn)

        return npv

    def _compute_rmse(self, **kwargs):
        """
        Computes root mean squared error

        Other Parameters
        ----------
        y_pred_new : np.ndarray
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
        y_pred_new : np.ndarray
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
        y_pred_new : np.ndarray
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

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_roc_auc_score : float
                The performance metric value
        """
        y_true = self.y_true[0]
        y_prob = self.y_prob[0]
        
        if y_prob is None:
            return None
        
        if 'y_pred_new' in kwargs:
            y_prob=kwargs['y_pred_new'][0]
        
        if self.sample_weight[0] is None :
            roc_auc = roc_auc_score(y_true=y_true, y_score=y_prob)
        else: 
            sample_weight = self.sample_weight[0]
            roc_auc = roc_auc_score(y_true=y_true, y_score=y_prob, sample_weight=self.sample_weight[0])

        return roc_auc

    def _compute_log_loss(self, **kwargs):
        """
        Computes the log loss score

        Other Parameters
        ----------
        y_pred_new : np.ndarray
            Copy of predicted targets as returned by classifier.

        Returns
        ----------
        _compute_log_loss : float
                The performance metric value
        """
        y_true = self.y_true[0]
        y_prob = self.y_prob[0]
        
        if y_prob is None:
            return None
        
        if 'y_pred_new' in kwargs:
            y_prob=kwargs['y_pred_new'][0]
            
        if self.sample_weight[0] is None :
            log_loss_score = log_loss(y_true=y_true, y_pred=y_prob)
        else: 
            sample_weight = self.sample_weight[0]
            log_loss_score = log_loss(y_true=y_true, y_pred=y_prob, sample_weight=self.sample_weight[0])

        return log_loss_score

    def _performance_dynamics(self):
        """
        Computes the dynamic performance metrics based on different threshold values

        Returns
        ----------
        d : dictionary
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
                for i in range(len(threshold)):
                    y_pred_new = [j > threshold[i] for j in self.y_prob[0]]  
                    y_pred_new = [int(elem) for elem in y_pred_new]  
                    tn, fp, fn, tp  = confusion_matrix(y_true=self.y_true[0], y_pred=y_pred_new).ravel()
                    d['selection_rate'] +=[(tp + fp) / (tp + fp + tn + fn)]

                    d['perf_metric_name'] = 'balanced_acc'
                    d['perf'] += [balanced_accuracy_score(y_true=self.y_true[0], y_pred=y_pred_new)]
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
        y_true: array
            Ground truth target values.

        y_prob : array of shape (n_samples, L), default=None
                Predicted probabilities as returned by classifier. For uplift models, L = 4. Else, L = 1.

        n_bins : int, default=10
            Number of equal-width bins in the range

        Returns
        ----------
        calibration_curve_bin : dictionary
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
        y_true: np.ndarray
            Ground truth target values.

        pos_label : array, default=1
            Label values which are considered favorable.
            For all model types except uplift, converts the favourable labels to 1 and others to 0.
            For uplift, user is to provide 2 label names e.g. [["a"], ["b"]] in fav label. The first will be mapped to treatment responded (TR) & second to control responded (CR).

        Returns
        ----------
        y_true_counts : dictionary
            Dictionary of proportion of classes
        """
        if self.label_size == -1:
            return None
        else:
            y_true_counts = pd.Series(y_true).value_counts(normalize = True)
            y_true_counts = y_true_counts.reset_index().replace({1: 'pos_label', 0:'neg_label'}).set_index('index')
            return y_true_counts[0].to_dict()

