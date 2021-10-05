import numpy as np
import sklearn.metrics as skm
from scipy import interpolate
import concurrent.futures

class ModelRateClassify:
    """
    Class to compute the interpolated base rates for classification models.
    """
    def __init__(self, y_true, y_prob):
        """
        Parameters
        -------------
        y_true: array of shape (n_samples,)
                Ground truth target values.

        y_prob: list of len = k of array of shape (n_samples, L)
                Predicted probabilities as returned by classifier. For non-uplift models, L = 4. Else, L = 1.
                K = 2 for uplift models, else K=1

        Instance Attributes
        -----------------------
        tpr: object
                Scipy interp1d object containing the function for true positive rate.

        fpr: object
                Scipy interp1d object containing the function for false positive rate.

        ppv: object
                Scipy interp1d object containing the function for precision score.

        forr: object
                Scipy interp1d object containing the function for false omission rate parity.

        selection_rate: object
                Scipy interp1d object containing the function for selection rate.

        base_selection_rate: object
                Scipy interp1d object containing the function for base selection rate.

        """
        (ths, tpr, fpr, ppv, forr, base_selection_rate, selection_rate) = ModelRateClassify.compute_rates(y_true, y_prob)
        self.tpr = interpolate.interp1d(ths, tpr)
        self.fpr = interpolate.interp1d(ths, fpr)
        self.ppv = interpolate.interp1d(ths, ppv)
        self.forr = interpolate.interp1d(ths, forr)
        self.selection_rate = interpolate.interp1d(ths, selection_rate)
        self.base_selection_rate = base_selection_rate

    def compute_rates(y_true, y_prob):
        """
        Computes the base rates for classification models.
        Parameters
        -------------
        y_true: array of shape (n_samples,)
                Ground truth target values.

        y_prob: list of len = k of array of shape (n_samples, L), default = None
                Predicted probabilities as returned by classifier. For non-uplift models, L = 4. Else, L = 1.
                K = 2 for uplift models, else K=1

        Returns
        ---------
        ths: array
                Array of size len(y_true) of threshold values equally binned between 0 and 1.

        tpr: array
                Array of size len(y_true) of true positive rate values

        fpr: array
                Array of size len(y_true) of false positive rate values.

        ppv: array
                Array of size len(y_true) of precision scores.

        forr: array
                Array of size len(y_true) of false omission rate parity values.

        selection_rate: array
                Array of size len(y_true) of selection rate values.

        base_selection_rate: array
                Array of size len(y_true) of base selection rate values.

        """
        fpr, tpr, ths = skm.roc_curve(y_true, y_prob, pos_label=1)
        ths[0] = 1.0  # roc_curve sets max threshold arbitrarily above 1
        ths = np.append(ths, [0.0])  # Add endpoints for ease of interpolation
        fpr = np.append(fpr, [1.0])
        tpr = np.append(tpr, [1.0])
        
        base_selection_rate = np.mean(y_true)
        base_reject_rate = 1 - base_selection_rate

        selection_rate = base_selection_rate * tpr + base_reject_rate * fpr
        reject_rate = 1 - selection_rate

        prob_tp = base_selection_rate * tpr
        ppv = np.divide(prob_tp, selection_rate, out=np.zeros_like(prob_tp), where=(selection_rate != 0))

        prob_fn0 = prob_tp * np.divide(1, tpr, out=np.zeros_like(prob_tp), where=(tpr != 0))
        prob_fn = np.where(tpr == 0, selection_rate, prob_fn0)
        forr = np.divide(prob_fn, reject_rate, out=np.zeros_like(prob_fn), where=(reject_rate != 0))

        return ths, tpr, fpr, ppv, forr, base_selection_rate, selection_rate



class ModelRateUplift:
    """
    Class to compute the interpolated base rates for uplift models.
    """
    def __init__(self, pred_outcome, e_lift, feature_mask, cost, revenue, proportion_of_interpolation_fitting, n_threads):

        """
        Parameters
        -------------
        pred_outcome : dictionary

        e_lift : float
                Empirical lift

        feature_mask : dictionary of lists
                Stores the mask array for every protected variable applied on the x_test dataset.

        cost: float
                Cost of the marketing treatment per customer

        revenue: float
                Revenue gained per customer

        proportion_of_interpolation_fitting : float

        n_threads : int

        Instance Attributes
        ---------------------
        harm: object
                Scipy interp1d object containing the function for rejected harm.

        profit: object
                Scipy interp1d object containing the function for profit.
        """

        self.n_threads = n_threads
        (ths, harm_array, profit_array) = self.compute_rates_uplift(pred_outcome, e_lift, feature_mask, cost, revenue, proportion_of_interpolation_fitting)
        
        self.harm = interpolate.interp1d(ths, harm_array)
        self.profit = interpolate.interp1d(ths, profit_array)
        

    def compute_rates_uplift(self, pred_outcome, e_lift, feature_mask, cost, revenue, proportion_of_interpolation_fitting):
        """
        Computes the base rates for uplift models.

        Parameters
        ------------------
        pred_outcome : dictionary


        e_lift : float
                Empirical lift

        feature_mask : dictionary of lists
                Stores the mask array for every protected variable applied on the x_test dataset.

        cost: float
                Cost of the marketing treatment per customer

        revenue: float
                Revenue gained per customer

        proportion_of_interpolation_fitting : float

        Returns
        -----------------
        ths: array
                Array of size len(y_true) of threshold values equally binned between -0.5 and 0.5.

        harm_array: array
                Array of size len(y_true) of rejected harm values

        profit_array: array
                Array of size len(y_true) of profit values.
        """

        harm_array = []
        profit_array = []

        ## define threshold bins
        #y_prob changed to y_prob_in Aug 18
        #ths = np.linspace(-0.5, 0.5, sum(feature_mask)) ### if size of y_prob is too large and affects performance, change to min(len(y_prob), 500)
        sum_feature_mask = sum(feature_mask)
        max_length = int(sum_feature_mask*proportion_of_interpolation_fitting)
        ths = np.linspace(e_lift.min(), e_lift.max(), max_length) 
        ths[-1] = e_lift.max()
        
        def compute_lift_per_thread(start,n_threads):
            
            harm_values_lst = []
            profit_values_lst = []
            
            for j in range(start,len(ths),n_threads):
                
                selection = np.array([i > j for i in e_lift])[feature_mask]
                pRejcT = pred_outcome['rej_treatment'][feature_mask][selection]
                pRejcC = pred_outcome['rej_control'][feature_mask][selection]
    
                pRcT = pred_outcome['acq_treatment'][feature_mask][selection]
                pRcC = pred_outcome['acq_control'][feature_mask][selection]                


                harm_values = sum(pRejcT - pRejcC) / len(selection) 
                profit_values = sum(pRcT * revenue - cost - pRcC * revenue)    

                harm_values_lst.append(harm_values)
                profit_values_lst.append(profit_values)

            return harm_values_lst, profit_values_lst

        threads = []
        n = len(ths)
        harm_array = np.zeros(n)
        profit_array = np.zeros(n)
        #with concurrent.futures.ProcessPoolExecutor() as executor: 
        with concurrent.futures.ThreadPoolExecutor(self.n_threads) as executor: 
            for i in range(self.n_threads):                                        
                threads.append(executor.submit(compute_lift_per_thread, i, self.n_threads))
            
            for i,thread in enumerate(threads):             
                res = thread.result()
                
                harm_array[i:n:self.n_threads] = res[0]
                profit_array[i:n:self.n_threads] = res[1]

        return ths, harm_array, profit_array
