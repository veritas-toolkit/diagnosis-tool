from ..util.utility import check_datatype, check_value, check_label
import numpy as np
import pandas as pd
from ..fairness.fairness import Fairness
from ..util.errors import *
from collections.abc import Iterable

class ModelContainer(object):

    """Helper class that holds the model parameters required for computations in all use cases."""
    def __init__(self, y_true, p_var, p_grp, model_type, model_name = 'auto', y_pred=None, y_prob=None, y_train=None, protected_features_cols=None, train_op_name="fit",
                 predict_op_name ="predict", feature_imp=None, sample_weight=None, pos_label=[[1]], neg_label=None, x_train=None, x_test=None, model_object=None):
        """
        Parameters
        ----------
        y_true : array of shape (n_samples,)
                Ground truth target values.

        p_var : list of size=k
                List of privileged groups within the protected variables.

        p_grp : dictionary of lists
                List of privileged groups within the protected variables.

        model_type : string
                The type of model to be evaluated, model_type is unique.

                Customer Marketing: "uplift" or "propensity" or "reject"
                Credit Scoring: "credit"
                Base Class: "default"

        Instance Attributes
        --------------------
        model_name : string, default="auto"
                Used to name the model artifact json file in compile function.

        y_pred : array of shape (n_samples,), default=None
                Predicted targets as returned by classifier.

        y_prob : array of shape (n_samples, L), default=None
                Predicted probabilities as returned by classifier. For uplift models, L = 4. Else, L = 1.
                y_prob column orders should be ["TR", "TN", "CR", "CN"] for uplift models.

        y_train : array of shape (m_samples,), default=None
                Ground truth for training data.

        protected_features_cols: pandas Dataframe of shape (n_samples, k), default=None
                This variable will be used for masking. If not provided, x_test will be used and x_test must be a pandas dataframe.

        train_op_name : string, default = "fit"
                The method used by the model_object to train the model. By default a sklearn model is assumed.

        predict_op_name : string, default = "predict"
                The method used by the model_object to predict the labels or probabilities. By default a sklearn model is assumed.
                For uplift models, this method should predict the probabilities and for non-uplift models it should predict the labels.

        feature_imp : pandas Dataframe of shape (n_features,2), default=None
                The feature importance computed on the model object fitted to x_train. Order of features must be the same as x_test & x_train.

        sample_weight : array of shape (n_samples,), default=None
                Used to normalize y_true & y_pred.

        pos_label : array, default = [[1]]
                Label values which are considered favorable.
                For all model types except uplift, converts the favourable labels to 1 and others to 0.
                For uplift, user is to provide 2 label names e.g. [["a"], ["b"]] in fav label. The first will be mapped to treatment responded (TR) & second to control responded (CR).

        neg_label : array, default = None
                Label values which are considered unfavorable.
                neg_label will only be used in uplift models.
                For uplift, user is to provide 2 label names e.g. [["c"], ["d"]] in unfav label. The first will be mapped to treatment rejected (TR) & second to control rejected (CR).

        x_train : pandas Dataframe of shape (m_samples, n_features) or string
                Training dataset. m_samples refers to number of rows in the training dataset. The string refers to the dataset path acceptable by the model (e.g. HDFS URI).
        
        x_test : pandas Dataframe or array of shape (n_samples, n_features) or string
                Testing dataset. The string refers to the dataset path acceptable by the model (e.g. HDFS URI).

        model_object : Object
                A blank model object used in the feature importance section for training and prediction.

        _input_validation_lookup : dictionary
                Contains the attribute, its correct data type and values (if it is applicable) for every argument passed by user.

        err : object
                VeritasError object to save errors

        pos_label2 : array, default=None
                Encoded pos_label value
        """
        self.x_train = x_train
        self.x_test = x_test
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.y_train = y_train
        self.p_var = p_var
        self.p_grp = p_grp
        self.protected_features_cols = protected_features_cols
        self.model_object = model_object
        self.train_op_name = train_op_name
        self.predict_op_name = predict_op_name
        self.feature_imp = feature_imp
        self.sample_weight = sample_weight
        self.model_name = model_name
        self.model_type = model_type
        if self.model_name == 'auto':
            self.model_name = model_type
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.pos_label2 = None

        self.err = VeritasError()

        self.check_label_length()
        
        check_y_label = None

        check_p_grp = None

        self.check_protected_columns()
        metric_group = None
        check_model_type = []
        for usecase in Fairness.__subclasses__():
            model_type_to_metric_lookup = (getattr(usecase, "_model_type_to_metric_lookup"))
            check_model_type = check_model_type + list(model_type_to_metric_lookup.keys())
            check_model_type = sorted(list(set(check_model_type)))
            if model_type in model_type_to_metric_lookup.keys():
                metric_group = model_type_to_metric_lookup[model_type][0]

        if y_true is not None and metric_group in ["uplift", "classification"]:
            check_y_label = list(set(y_true))
        if p_var is not None :
            check_p_grp=dict.fromkeys(p_var)

        # Dictionary for input data
        if protected_features_cols is None:
            if type(self.x_test) == pd.DataFrame:
                self.protected_features_cols = self.x_test[p_var]
            else:
                self.err.push('type_error', var_name='x_test', given=str(type(str)), expected=str(pd.DataFrame),
                            function_name='ModelContainer')
                self.err.pop()

        for var in check_p_grp.keys():
            check_p_grp[var] = self.protected_features_cols[var].unique()
            
        NoneType = type(None)

        # Dictionary for expected data type
        # if value range is a tuple, will be considered as a numerical range
        # if value range is a list/set of str, will be considered as a collection of available values
        self._input_validation_lookup = {
            "y_true": [(list, np.ndarray, pd.Series), ],
            "y_train": [(NoneType, list, np.ndarray, pd.Series), ], 
            "y_pred": [(NoneType, list, np.ndarray, pd.Series), check_y_label], 
            "y_prob": [(NoneType, list, np.ndarray, pd.Series, pd.DataFrame), (-0.01,1.01)],
            "p_var":  [(list,), str],
            "protected_features_cols": [(NoneType, pd.DataFrame), self.p_var],
            "p_grp":  [(dict,), check_p_grp], 
            "x_train": [(NoneType, pd.DataFrame, str), None],
            "x_test":  [(NoneType, pd.DataFrame, str), None],
            "train_op_name": [(str,), None],
            "predict_op_name": [(str,), None],
            "feature_imp":     [(NoneType, pd.DataFrame), (np.dtype('O'), np.dtype('float64'))], 
            "sample_weight":   [(NoneType, list, np.ndarray), (0, np.inf)], 
            "model_name":      [(str,), None],
            "model_type":      [(str,), check_model_type],
            "pos_label":       [(list,), check_y_label],
            "neg_label":       [(NoneType, list), check_y_label]
        }
        
        check_datatype(self)
        
        #if model name is longer than 20 characters, will keep the first 20 only
        self.model_name = self.model_name[0:20]

        #convert to numpy array for easier processing
        self.y_true = np.array(self.y_true)
        if self.y_pred is not None :
            self.y_pred = np.array(self.y_pred)
        if self.y_prob is not None :            
            self.y_prob = np.array(self.y_prob)
        if self.y_train is not None :
            self.y_train = np.array(self.y_train)
        if self.sample_weight is not None :
            self.sample_weight = np.array(self.sample_weight)

        check_value(self)

        self.check_data_consistency()
        if metric_group in ["uplift","classification"]:
            self.check_label_consistency()
    
            if len(self.y_true.shape) == 1 and self.y_true.dtype.kind in ['i','O','U']:
                self.y_true, self.pos_label2 = check_label(self.y_true, self.pos_label, self.neg_label)  
            if self.y_pred is not None and len(self.y_pred.shape) == 1 and self.y_pred.dtype.kind in ['i','O','U']:
                self.y_pred, self.pos_label2 = check_label(self.y_pred, self.pos_label, self.neg_label)
                
            if self.y_train is not None and len(self.y_train.shape) == 1 and self.y_train.dtype.kind in ['i','O','U']:
                self.y_train, self.pos_label2 = check_label(self.y_train, self.pos_label, self.neg_label)

    def check_protected_columns(self):     
        """
        Check p_var, x_test and protected_feature_columns consistency

        Returns:
        ---------------
        successMsg : string
            If there are no errors, a success message will be returned     
        """
        err_ = []
        successMsg = "protected column check completed without issue"

        if type(self.p_var) != list:
            err_.append(['type_error', "p_var", type(self.p_var), list])

        elif self.protected_features_cols is None and self.x_test is None :
            err_.append(['length_error', "protected_features_cols and x_test", "None for both", "not both are None"])
        
        elif self.protected_features_cols is not None :
            if sum([x in self.p_var for x in self.protected_features_cols.columns]) != len(self.p_var): 
                err_.append(['value_error', "p_var", str(self.p_var), str(list(self.protected_features_cols.columns))])
        elif self.protected_features_cols is None and self.x_test is not None :           
            if sum([x in self.p_var for x in self.x_test.columns]) != len(self.p_var) : 
                err_.append(['value_error', "p_var", str(self.p_var), str(list(self.x_test.columns))])
          
        if err_ == []:
            return successMsg
        else:
            for i in range(len(err_)):
                self.err.push(err_[i][0], var_name=err_[i][1], given=err_[i][2], expected=err_[i][3],
                            function_name="check_protected_columns")
            self.err.pop()

    def check_data_consistency(self):
        """
        Check rows and columns are of consistent size across the various datasets and the number & match of the unique values.

        Returns:
        ---------------
        successMsg : string
            If there are no errors, a success message will be returned       
        """
        err_ = []

        successMsg = "data consistency check completed without issue"

        test_row_count = self.y_true.shape[0] 

        # check protected_features_cols
        # check cols of protected_features_cols is same as p_var
        pro_f_cols_rows = len(self.protected_features_cols.index)
        pro_f_cols_cols = len(self.protected_features_cols.columns)
        if pro_f_cols_rows != test_row_count:
            err_.append(['length_error', "protected_features_cols row", str(pro_f_cols_rows), str(test_row_count)])
        if pro_f_cols_cols != len(self.p_var):
            err_.append(['length_error', "p_var array", str(len(self.p_var)), str(pro_f_cols_cols)])

        #check train datasets if y_train is not None
        if self.y_train is not None:
            train_row_count = self.y_train.shape[0] 
            # check x_train
            if type(self.x_train) == pd.DataFrame:
                x_train_rows = len(self.x_train.index)
                if x_train_rows != train_row_count:
                    x_train = None

        #y_prob should be float
        if self.y_prob is not None and self.y_prob.dtype.kind != 'f':
            self.err.push('type_error', var_name="y_prob", given= "not type float64", expected="float64", function_name="check_data_consistency")

        # if both x_test and x_train are df, check they have same no. of columns
        if type(self.x_train) == pd.DataFrame and type(self.x_test) == pd.DataFrame:
            x_train_cols = len(self.x_train.columns)
            x_test_cols = len(self.x_test.columns)
            if x_train_cols != x_test_cols:
                err_.append(['length_error', "x_train column", str(x_train_cols), str(x_test_cols)])

        #check pos_label size and neg_label size
        for usecase in Fairness.__subclasses__():
            model_type_to_metric_lookup = (getattr(usecase, "_model_type_to_metric_lookup"))
            if self.model_type in model_type_to_metric_lookup.keys():
                    #assumption: model_type is unique across all use cases
                label_size = model_type_to_metric_lookup.get(self.model_type)[2]
                
        # if label size requirement is -1/2, no need to check
        if label_size > 0:
            if self.neg_label is None and self.model_type == "uplift":
                err_.append(['value_error', "neg_label", "None", "not None"])
            if self.neg_label is not None:
                neg_label_size = len(self.neg_label)
                if neg_label_size != label_size:
                    err_.append(['length_error', "neg_label", str(neg_label_size), str(label_size)])
            pos_label_size = len(self.pos_label)
            if pos_label_size != label_size:
                err_.append(['length_error', "pos_label", str(pos_label_size), str(label_size)])

        #y_pred and y_prob should not be both none
        if self.y_pred is None and self.y_prob is None:
            err_.append(['length_error', "y_pred and y_prob", "None for both", "not both are None"])

        # y_true should 1 columns
        # y_true, y_pred, sample weight, are in same shape
        #Based on the length of pos_label, if 1, the y_prob will be nx1
        #Based on the length of pos_label, if 2, the y_prob will be nx4
        y_true_shape = self.y_true.shape
        if len(y_true_shape) == 1:
            y_true_shape = (y_true_shape[0], 1)
        
        check_list = ["y_pred","y_prob", "sample_weight"]
        check_order = ["row","column"]
        check_dict = {}
        for var_name in check_list:
            var_value = getattr(self, var_name)
            if var_value is not None:
                if type(var_value) == list:
                    var_value = np.array(var_value)
                var_shape = var_value.shape
                for i in range(len(check_order)):
                    try:
                        given_size = var_shape[i]
                    except IndexError:
                        given_size = 1
                    expected_size = y_true_shape[i]
                    
                    #Based on the length of pos_label, if 1, the y_prob will be nx1
                    #Based on the length of pos_label, if 2, the y_prob will be nx4
                    if var_name =="y_prob" and check_order[i] == "column" and label_size == 2:
                        expected_size = 4
                        
                    if given_size != expected_size:
                        err_.append(['length_error', var_name + " " + check_order[i], str(given_size), str(expected_size)])

        if err_ == []:
            return successMsg
        else:
            for i in range(len(err_)):
                self.err.push(err_[i][0], var_name=err_[i][1], given=err_[i][2], expected=err_[i][3],
                            function_name="check_data_consistency")
            self.err.pop()

    def check_label_consistency(self):
        """
        Checks if the labels and values in y_pred, y_train and y_true are consistent

        Returns:
        ---------------
        successMsg : string
            If there are no errors, a success message will be returned   
        """
        err_ = []
        successMsg = "data consistency check completed without issue"

        if self.y_pred is not None :
            if set(self.y_true) != set(self.y_pred):
                err_.append(['value_error', "y_pred labels", str(set(self.y_pred)), str(set(self.y_true))])
        
        if self.y_train is not None :
            if set(self.y_true) != set(self.y_train):
                self.y_train = None
        if err_ == []:
            return successMsg
        else:
            for i in range(len(err_)):
                self.err.push(err_[i][0], var_name=err_[i][1], given=err_[i][2], expected=err_[i][3],
                            function_name="check_label_consistency")            
            self.err.pop()

    def check_label_length(self):
        """
        Checks if the length of labels is correct according to the model_type
        
        Returns:
        ---------------
        successMsg : string
            If there are no errors, a success message will be returned   
        """
        err_ = []
        successMsg = "label length check completed without issue"
        for usecase in Fairness.__subclasses__():
            model_type_to_metric_lookup = (getattr(usecase, "_model_type_to_metric_lookup"))
            if self.model_type in model_type_to_metric_lookup.keys():
                min_label_size = model_type_to_metric_lookup.get(self.model_type)[1]
                pos_label_size = model_type_to_metric_lookup.get(self.model_type)[2]
                if min_label_size != -1 :
                    if self.pos_label is not None :
                        if len(self.pos_label) != pos_label_size :
                            err_.append(["length_error", "pos_label length", len(self.pos_label), str(pos_label_size)])
                    if self.y_true is not None :
                        if len(set(self.y_true))<min_label_size :
                            err_.append(["length_error", "y_true label length", len(set(self.y_true)), '>='+str(min_label_size)])
                    if self.y_train is not None :
                        if len(set(self.y_train))<min_label_size :
                            err_.append(["length_error", "y_train label length", len(set(self.y_train)), '>='+str(min_label_size)])
                    if self.y_pred is not None :
                        if len(set(self.y_pred))<min_label_size :
                            err_.append(["length_error", "y_pred label length", len(set(self.y_pred)), '>='+str(min_label_size)])

        if err_ == []:
            return successMsg
        else:
            for i in range(len(err_)):
                self.err.push(err_[i][0], var_name=err_[i][1], given=err_[i][2], expected=err_[i][3],
                            function_name="check_label_length")            
            self.err.pop()
                    
    def clone(self, y_true, model_object, y_pred=None, y_prob=None, y_train=None, train_op_name="fit",
                 predict_op_name ="predict", feature_imp=None, sample_weight=None,  pos_label=[[1]], neg_label=None):

        """
        Clone ModelContainer object

        Parameters
        ---------------
        y_true : array of shape (n_samples,)
                Ground truth target values.

        model_object : Object
                A blank model object used in the feature importance section for training and prediction.

        y_pred : array of shape (n_samples,), default=None
                Predicted targets as returned by classifier.

        y_prob : array of shape (n_samples, L), default=None
                Predicted probabilities as returned by classifier. For uplift models, L = 4. Else, L = 1.
                y_prob column orders should be ["TR", "TN", "CR", "CN"] for uplift models.

        y_train : array of shape (m_samples,), default=None
                Ground truth for training data.

        train_op_name : string, default = "fit"
                The method used by the model_object to train the model. By default a sklearn model is assumed.

        predict_op_name : string, default = "predict"
                The method used by the model_object to predict the labels or probabilities. By default a sklearn model is assumed.
                For uplift models, this method should predict the probabilities and for non-uplift models it should predict the labels.

        feature_imp : pandas Dataframe of shape (n_features,2), default=None
                The feature importance computed on the model object fitted to x_train. Order of features must be the same as x_test & x_train.

        sample_weight : array of shape (n_samples,), default=None
                Used to normalize y_true & y_pred.

        pos_label : array, default = [[1]]
                Label values which are considered favorable.
                For all model types except uplift, converts the favourable labels to 1 and others to 0.
                For uplift, user is to provide 2 label names e.g. [["a"], ["b"]] in fav label. The first will be mapped to treatment responded (TR) & second to control responded (CR).

        neg_label : array, default = None
                Label values which are considered unfavorable.
                neg_label will only be used in uplift models.
                For uplift, user is to provide 2 label names e.g. [["c"], ["d"]] in unfav label. The first will be mapped to treatment rejected (TR) & second to control rejected (CR).

        Returns
        ----------------
        clone_obj : object
        """
        clone_obj = ModelContainer(y_true = y_true, p_var = self.p_var, p_grp = self.p_grp, x_train= self.x_train,  x_test = self.x_test, model_object = model_object, model_type = self.model_type, model_name = "clone", y_pred=y_pred, y_prob=y_prob, y_train = y_train, protected_features_cols=self.protected_features_cols, train_op_name=train_op_name,
                 predict_op_name = predict_op_name, feature_imp=self.feature_imp, sample_weight=self.sample_weight, pos_label=pos_label, neg_label=neg_label)
        
        return clone_obj

