import numpy as np
import sklearn.metrics as skm
from .fairness_metrics import FairnessMetrics
from .modelrates import *
from ..config.constants import Constants

class TradeoffRate(object):
    """
    Class to compute the tradeoff between performance and fairness metrics across a range of threshold values.
    """
    def __init__(self, usecase_obj):
        """
        Parameters
        ----------
        usecase_obj : object
                Object is initialised in use case classes.

        Instance attributes
        -----------
        msg : string, default=None
                Message 

        perf_metric_name : string 
                Name of the primary performance metric to be used for computations.

        fair_metric_name : string
                Name of the primary fairness metric to be used for computations.

        metric_group : string
                Type of fair_metric

        p_var : list
                List of protected variables used for fairness analysis.

        curr_p_var: string
                Current protected variable

        result : dictionary, default=None
                Stores the results of the computations. Refer to the sample JSON artifact for an example.

        fair_neutral_tolerance : float
                Tolerance value between 0 and 0.1 (inclusive) used in the performance-fairness tradeoff analysis section to filter the primary fairness metric values.

        proportion_of_interpolation_fitting : float
                Proportion of interpolation fitting

        y_true : array of shape (n_samples,)
                Ground truth target values.

        y_prob : array of shape (n_samples, L), default=None
                Predicted probabilities as returned by classifier. For uplift models, L = 4. Else, L = 1.

        sample_weight : array of shape (n_samples,)
                Used to normalize y_true & y_pred.

        feature_mask : dictionary of lists
                Stores the mask array for every protected variable applied on the x_test dataset.

        map_metric_to_method : dictionary
                Mapping of metric name to respective compute function
        """
        self.msg = None
        self.perf_metric_name = usecase_obj.perf_metric_name
        self.fair_metric_name = usecase_obj.fair_metric_name
        self.metric_group = FairnessMetrics.map_fair_metric_to_group[self.fair_metric_name][1]
        if self.metric_group == "uplift":
            if usecase_obj.spl_params['revenue'] is not None and usecase_obj.spl_params['treatment_cost'] is not None:                
                if self.perf_metric_name not in ["expected_profit", "emp_lift"]:
                    self.perf_metric_name = "expected_profit" 
            else :
                self.perf_metric_name = "emp_lift"		

            self.fair_metric_name = "rejected_harm"
        elif self.metric_group == "classification":
            self.perf_metric_name = usecase_obj.perf_metric_name
            if self.perf_metric_name not in ["balanced_acc", "f1_score"]:
                self.perf_metric_name = "balanced_acc"
            is_valid_metric = FairnessMetrics.map_fair_metric_to_group[self.fair_metric_name][3]
            if is_valid_metric == False:
                self.fair_metric_name = usecase_obj._select_fairness_metric_name() #usecase._select_fairness_metric_name() get the usecase object over

        self.p_var = usecase_obj.model_params[0].p_var
        self.curr_p_var = None
        self.result = None
        self.fair_neutral_tolerance = usecase_obj.fair_neutral_tolerance
        
        if self.metric_group == "uplift":
            self.proportion_of_interpolation_fitting = usecase_obj.proportion_of_interpolation_fitting
        if usecase_obj.model_params[0].model_type == "uplift" or usecase_obj.model_params[0].model_type == "credit":
            self.spl_params = usecase_obj.spl_params
        if self.metric_group == "uplift": 
            self.y_true = [model.y_true for model in usecase_obj.model_params]
            self.y_prob = [model.y_prob for model in usecase_obj.model_params]
            self.pred_outcome = usecase_obj.pred_outcome
            self.e_lift = usecase_obj.e_lift           
        else:
            self.y_true = np.array(usecase_obj.model_params[0].y_true)
            self.y_prob = np.array(usecase_obj.model_params[0].y_prob)
            self.sample_weight = np.array(usecase_obj.model_params[0].sample_weight) if usecase_obj.model_params[0].sample_weight is not None else None
        self.feature_mask = usecase_obj.feature_mask

        self.map_metric_to_method = {
        'balanced_acc': self._compute_bal_accuracy_grid,
        'f1_score': self._compute_f1_grid,
        'equal_opportunity': self._compute_equal_opportunity_tr,
        'disparate_impact': self._compute_disparate_impact_tr,
        'demographic_parity': self._compute_demographic_parity_tr,
        'for_parity': self._compute_false_omission_rate_parity_tr,
        'fdr_parity': self._compute_false_discovery_rate_parity_tr,
        'ppv_parity': self._compute_positive_predictive_parity_tr,
        'npv_parity': self._compute_negative_predictive_parity_tr,
        'tnr_parity': self._compute_tnr_parity_tr,
        'fnr_parity': self._compute_fnr_parity_tr,
        'fpr_parity': self._compute_fpr_parity_tr,
        'equal_odds': self._compute_equalized_odds_tr,
        'neg_equal_odds': self._compute_negative_equalized_odds_tr,
        'calibration_by_group': self._compute_calibration_by_group_tr,
        'rejected_harm': self._compute_rejected_harm_tr,
        'expected_profit': self._compute_expected_profit_tr,
        'emp_lift': self._compute_emp_lift_tr
        }

    def compute_tradeoff(self, n_threads, tdff_pbar):
        """
        Computes the tradeoff values.

        Parameters
        ------------
        n_threads : int
                Number of currently active threads of a job

        tdff_pbar : 
                Progress bar

        Returns
        ------------
        result : dictionary
                Stores the results of the computations.
                    - fairness metric name
                    - performance metric name
                    - fairness metric values
                    - performance metric values
                    - threshold_x values
                    - threshold_y values
                    - max_perf_point
                    - max_perf_single_th
                    - max_perf_neutral_fair
        """
        self.result = {}
        # based on model type, compute the base rates accordingly
        if self.metric_group == "uplift":

            # define meshgrid
            self.th_x = np.linspace(min(self.e_lift)*0.7, max(self.e_lift)*0.7, Constants().tradeoff_threshold_bins)
            self.th_y = np.linspace(min(self.e_lift)*0.7, max(self.e_lift)*0.7, Constants().tradeoff_threshold_bins)
            self.th_a, self.th_b = np.meshgrid(self.th_x, self.th_y, sparse=True)

            #access from the special params
            cost = self.spl_params["treatment_cost"]
            revenue = self.spl_params["revenue"]
            tdff_pbar.update(5)
            prog = round(80/(len(self.p_var)), 2)

            ## iterate through the protected_features
            for i in self.p_var:
                self.curr_p_var = i
                self.uplift_rates_a = ModelRateUplift(self.y_true, self.pred_outcome, self.e_lift, self.feature_mask[i], cost, revenue, self.proportion_of_interpolation_fitting, n_threads) 
                self.uplift_rates_b = ModelRateUplift(self.y_true, self.pred_outcome, self.e_lift, ~self.feature_mask[i], cost, revenue, self.proportion_of_interpolation_fitting, n_threads)
                #### compute the metrics and create dictionary
                fair_values = self.map_metric_to_method[self.fair_metric_name]()
                perf_values = self.map_metric_to_method[self.perf_metric_name]()

                ### add metric and threshold values to result dictionary
                self.result[i]= {}
                self.result[i]['fair_metric_name'] = self.fair_metric_name
                self.result[i]['perf_metric_name'] = self.perf_metric_name

                self.result[i]['perf'] = perf_values

                self.result[i]['fair'] = fair_values

                self.result[i]['th_x'] = self.th_x
                self.result[i]['th_y'] = self.th_y

                best_th1, best_th2, best_th3 = TradeoffRate._compute_max_perf(self,perf_values,fair_values)
                self.result[i]['max_perf_point'] = best_th2
                self.result[i]['max_perf_single_th'] = best_th1
                self.result[i]['max_perf_neutral_fair'] = best_th3
                tdff_pbar.update(prog)

        elif self.metric_group == "classification":
            # base rates for entire dataset to calculate the best performance metric grid used in _compute_max_perf()
            self.ths, self.tpr, self.fpr, self.ppv, self.forr, self.base_selection_rate, self.selection_rate = ModelRateClassify.compute_rates(self.y_true, self.y_prob, self.sample_weight)

            # define meshgrid
            self.th_x = np.linspace(Constants().classify_min_threshold, Constants().classify_max_threshold, Constants().tradeoff_threshold_bins)
            self.th_y = np.linspace(Constants().classify_min_threshold, Constants().classify_max_threshold, Constants().tradeoff_threshold_bins)

            self.th_a, self.th_b = np.meshgrid(self.th_x, self.th_y, sparse=True)
            tdff_pbar.update(5)
            prog = round(80/(len(self.p_var)), 2)

            # iterate through the protected_features
            for i in self.p_var:
                self.curr_p_var = i
                # initialising ModelRates object to access interpolated base rates
                if self.sample_weight is None :
                    self.rates_a = ModelRateClassify(self.y_true[self.feature_mask[i]], self.y_prob[self.feature_mask[i]])
                    self.rates_b = ModelRateClassify(self.y_true[~self.feature_mask[i]], self.y_prob[~self.feature_mask[i]])                
                else :
                    self.rates_a = ModelRateClassify(self.y_true[self.feature_mask[i]], self.y_prob[self.feature_mask[i]], self.sample_weight[self.feature_mask[i]])
                    self.rates_b = ModelRateClassify(self.y_true[~self.feature_mask[i]], self.y_prob[~self.feature_mask[i]], self.sample_weight[~self.feature_mask[i]])

                # base rates for privileged and unprivileged groups to be used in classification based compute functions
                self.tpr_a, self.tpr_b, self.fpr_a, self.fpr_b, self.selection_rate_a, self.selection_rate_b, self.forr_a, self.forr_b, self.ppv_a, self.ppv_b = self.rates_a.tpr(self.th_a), self.rates_b.tpr(self.th_b), self.rates_a.fpr(self.th_a), self.rates_b.fpr(self.th_b), self.rates_a.selection_rate(self.th_a), self.rates_b.selection_rate(self.th_b), self.rates_a.forr(self.th_a), self.rates_b.forr(self.th_b), self.rates_a.ppv(self.th_a), self.rates_b.ppv(self.th_b)

                # compute the metrics and create dictionary
                fair_values = self.map_metric_to_method[self.fair_metric_name]()

                # add metric and threshold values to result dictionary
                self.result[i]= {}

                self.result[i]['fair_metric_name'] = self.fair_metric_name
                self.result[i]['perf_metric_name'] = self.perf_metric_name

                perf_values = self.map_metric_to_method[self.perf_metric_name]()

                self.result[i]['fair'] = fair_values
                self.result[i]['perf'] = perf_values

                self.result[i]['th_x'] = self.th_x
                self.result[i]['th_y'] = self.th_y

                best_th1, best_th2, best_th3 = TradeoffRate._compute_max_perf(self,perf_values,fair_values)
                self.result[i]['max_perf_point'] = best_th2
                self.result[i]['max_perf_single_th'] = best_th1
                self.result[i]['max_perf_neutral_fair'] = best_th3
                tdff_pbar.update(prog)
        else :
            self.result = None
            self.msg = "Tradeoff has been skipped due to fair_metric_name"
            return

    @staticmethod
    def _compute_max_perf(self,perf_grid,fair_grid):
        """
        Returns the max perf point, max perf single th and max perf neutral fair.
        
        Parameters
        ---------
        perf_grid :

        fair_grid :

        Returns
        ---------
        best_th1 : list
                Best single performance metric and associated threshold
        
        best_th2 : list
                Best single performance metric (split) and associated threshold
        
        best_th3 : list
                Best fairness-constrained performance metric and associated threshold
        """
        # compute the best single performance metric and associated threshold
        best_single_perf = np.max(np.fliplr(perf_grid).diagonal())
        best_single_th = self.th_x[np.argmax(np.fliplr(perf_grid).diagonal())]
        best_th1 = [best_single_th, best_single_th, best_single_perf]

        # compute the best single performance metric (split) and associated threshold
        idx = np.unravel_index(perf_grid.argmax(), perf_grid.shape)
        best_split_th_a, best_split_th_b = self.th_x[idx[1]], self.th_y[idx[0]]
        best_split_perf = perf_grid.max()
        best_th2 = [best_split_th_a, best_split_th_b, best_split_perf]

        # compute the best fairness-constrained performance metric and associated threshold
        constrained_grid = np.copy(perf_grid)
        fair_neutral_pos = FairnessMetrics.map_fair_metric_to_group.get(self.fair_metric_name)[2]
        if fair_neutral_pos == "ratio":
            fair_neutral_pos = 1
        else:
            fair_neutral_pos = 0

        while (np.absolute(fair_grid - fair_neutral_pos) > self.fair_neutral_tolerance).all():
            self.fair_neutral_tolerance *= 2

        constrained_grid[np.where(np.absolute(fair_grid - fair_neutral_pos) > self.fair_neutral_tolerance)] = 0
        self.fair_grid = fair_grid
        idx = np.unravel_index(constrained_grid.argmax(), constrained_grid.shape)
        best_con_th_a, best_con_th_b = self.th_x[idx[1]], self.th_y[idx[0]]
        best_con_perf = constrained_grid.max()
        best_th3 = [best_con_th_a, best_con_th_b, best_con_perf]

        return best_th1, best_th2, best_th3

    def _compute_bal_accuracy_grid(self):
        """
        Compute balanced accuracy grid

        Returns
        --------
        bal_acc : list of lists of floats
                Grid of balanced accuracy values
        """
        tpr_a, tpr_b = self.rates_a.tpr(self.th_a), self.rates_b.tpr(self.th_b)
        fpr_a, fpr_b = self.rates_a.fpr(self.th_a), self.rates_b.fpr(self.th_b)
        mask = self.feature_mask[self.curr_p_var]
        # Performance
        # Combine TPRs: P(R=1|Y=1) = P(R=1|Y=1,A=1)P(A=1|Y=1) + P(R=1|Y=1,A=0)P(A=0|Y=1)
        tpr = (tpr_a * np.mean(mask[self.y_true == 1]) +
               tpr_b * np.mean(~mask[self.y_true == 1])) ####self.y_true[0]
        # Combine FPRs: P(R=1|Y=0) = P(R=1|Y=0,A=1)P(A=1|Y=0) + P(R=1|Y=0,A=0)P(A=0|Y=0)
        fpr = (fpr_a * np.mean(mask[self.y_true == 0]) +
               fpr_b * np.mean(~mask[self.y_true == 0]))
        bal_acc = 0.5 * (tpr + 1 - fpr)
        return bal_acc

    def _compute_f1_grid(self):
        """
        Compute F1 score grid

        Returns
        --------
        f1 : list of lists of floats
                Grid of f1 score values
        """
        tpr_a, tpr_b = self.rates_a.tpr(self.th_a), self.rates_b.tpr(self.th_b)
        ppv_a, ppv_b = self.rates_a.ppv(self.th_a), self.rates_b.ppv(self.th_b)
        
        mask = self.feature_mask[self.curr_p_var]
        # Performance
        # Combine TPRs: P(R=1|Y=1) = P(R=1|Y=1,A=1)P(A=1|Y=1) + P(R=1|Y=1,A=0)P(A=0|Y=1)
        tpr = (tpr_a * np.mean(mask[self.y_true == 1]) +
               tpr_b * np.mean(~mask[self.y_true == 1]))
        # Combine FPRs: P(R=1|Y=0) = P(R=1|Y=0,A=1)P(A=1|Y=0) + P(R=1|Y=0,A=0)P(A=0|Y=0)
        ppv = (ppv_a * [np.mean(mask[self.y_prob > th]) for th in self.th_a.ravel()] ) + \
                (ppv_b * [np.mean(~mask[self.y_prob >th]) for th in self.th_b.ravel()] )
        
        f1 = 2 * ((ppv*tpr)/(ppv+tpr))
        return f1

    def _compute_equal_opportunity_tr(self):
        """
        Computes the difference in equal opportunity

        Returns
        ---------
        _compute_equal_opportunity_tr : float
                tradeoff value
        """
        return self.tpr_a - self.tpr_b

    def _compute_disparate_impact_tr(self):
        """
        Computes the ratio of approval rate between the privileged and unprivileged groups

        Returns
        ---------
        _compute_disparate_impact_tr : float
                tradeoff value
        """
        return self.selection_rate_a / self.selection_rate_b

    def _compute_demographic_parity_tr(self):
        """
        Computes the difference in approval rate between the privileged and unprivileged groups

        Returns
        ---------
        _compute_demographic_parity_tr : float
                tradeoff value
        """
        return self.selection_rate_a - self.selection_rate_b

    def _compute_false_omission_rate_parity_tr(self):
        """
        Computes the difference in negative predictive values between the privileged and unprivileged groups

        Returns
        ---------
        _compute_false_omission_rate_parity_tr : float
                tradeoff value
        """
        return self.forr_a - self.forr_b

    def _compute_false_discovery_rate_parity_tr(self):
        """
        Computes the difference in false discovery rate values between the privileged and unprivileged groups

        Returns
        ---------
        _compute_false_discovery_rate_parity_tr : float
                tradeoff value
        """
        return self.ppv_b - self.ppv_a

    def _compute_positive_predictive_parity_tr(self):
        """
        Computes the difference in positive predictive values between the privileged and unprivileged groups

        Returns
        ---------
        _compute_positive_predictive_parity_tr : float
                tradeoff value
        """
        return self.ppv_a - self.ppv_b

    def _compute_negative_predictive_parity_tr(self):
        """
        Computes the difference in negative predictive values between the privileged and unprivileged groups
        
        Returns
        ---------
        None
        """
        return self.forr_b - self.forr_a

    def _compute_tnr_parity_tr(self):
        """
        Computes the difference in true negative rates between the privileged and unprivileged groups

        Returns
        ---------
        _compute_fnr_parity_tr : float
                tradeoff value
        """
        return self.fpr_b - self.fpr_a

    def _compute_fnr_parity_tr(self):
        """
        Computes the difference in false negative rates between the privileged and unprivileged groups

        Returns
        ---------
        _compute_fnr_parity_tr : float
                tradeoff value
        """
        return self.tpr_b - self.tpr_a

    def _compute_fpr_parity_tr(self):
        """
        Computes the difference in false positive rates between the privileged and unprivileged groups

        Returns
        ---------
        _compute_fpr_parity_tr : float
                tradeoff value
        """
        return self.fpr_a - self.fpr_b

    def _compute_equalized_odds_tr(self):
        """
        Computes the difference in equalized odds between the privileged and unprivileged groups 

        Returns
        ---------
        _compute_equalized_odds_tr : float
                tradeoff value
        """
        return ((self.tpr_a - self.tpr_b) + (self.fpr_a - self.fpr_b))/2

    def _compute_negative_equalized_odds_tr(self):
        """
        Computes the difference in negative equalized odds between the privileged and unprivileged groups 

        Returns
        ---------
        _compute_negative_equalized_odds_tr : float
                tradeoff value
        """
        return ((self.fpr_b - self.fpr_a) + (self.tpr_b - self.tpr_a))/2 

    def _compute_calibration_by_group_tr(self):
        """
        Computes the difference in calibration by group between the privileged and unprivileged groups

        Returns
        ---------
        _compute_calibration_by_group_tr : float
                tradeoff value
        """
        return ((self.ppv_a - self.ppv_b) + (self.forr_a - self.forr_b)) / 2

    def _compute_rejected_harm_tr(self):
        """
        Computes the difference in harm from rejection between treatment and control groups

        Returns
        ---------
        _compute_rejected_harm_tr : float
                tradeoff value
        """
        return self.uplift_rates_a.harm(self.th_a) - self.uplift_rates_b.harm(self.th_b)
        
    def _compute_expected_profit_tr(self):
        """
        Computes the total expected profit for given threshold

        Returns
        ---------
        _compute_expected_profit_tr : float
                tradeoff value
        """
        return self.uplift_rates_a.profit(self.th_a) + self.uplift_rates_b.profit(self.th_b)

    def _compute_emp_lift_tr(self):
        """
        Computes the difference in empirical lift for given threshold

        Returns
        ---------
        _compute_emp_lift_tr : float
                tradeoff value
        """
        mask = self.feature_mask[self.curr_p_var]
        y_true_a = self.y_true[1][mask]
        tr_p = np.array([sum(y_true_a[self.e_lift[mask]> th] =='TR') for th in self.th_a.ravel()])
        tn_p = np.array([sum(y_true_a[self.e_lift[mask]> th] =='TN') for th in self.th_a.ravel()])
        cr_p = np.array([sum(y_true_a[self.e_lift[mask]> th] =='CR') for th in self.th_a.ravel()])
        cn_p = np.array([sum(y_true_a[self.e_lift[mask]> th] =='CN') for th in self.th_a.ravel()])
        t_p = np.add(tr_p, tn_p)
        c_p = np.add(cr_p, cn_p)

        y_true_b = self.y_true[1][~mask]
        tr_u = np.array([sum(y_true_b[self.e_lift[~mask]> th] =='TR') for th in self.th_b.ravel()])
        tn_u = np.array([sum(y_true_b[self.e_lift[~mask]> th] =='TN') for th in self.th_b.ravel()])
        cr_u = np.array([sum(y_true_b[self.e_lift[~mask]> th] =='CR') for th in self.th_b.ravel()])
        cn_u = np.array([sum(y_true_b[self.e_lift[~mask]> th] =='CN') for th in self.th_b.ravel()])
        t_u = np.add(tr_u, tn_u)
        c_u = np.add(cr_u, cn_u)

        treatment_response_rate = np.add((self.uplift_rates_a.emp_lift_tr(self.th_a))*t_p, self.uplift_rates_b.emp_lift_tr(self.th_b)*t_u)/(t_p + t_u)
        control_response_rate = np.add((self.uplift_rates_a.emp_lift_cn(self.th_a))*c_p, self.uplift_rates_b.emp_lift_cn(self.th_b)*c_u)/(c_p + c_u)
        
        return treatment_response_rate-control_response_rate