#import ModelWrapper
from .custom.modelwrapper import ModelWrapper
from .utility import check_datatype, check_value, check_label
import numpy as np
import pandas as pd
from .fairness.fairness import Fairness
from .fairness.credit_scoring import CreditScoring
from .fairness.customer_marketing import CustomerMarketing
from .ErrorCatcher import VeritasError
import sys

class ModelContainer(object):
    """Helper class that holds the model parameters required for computations in all use cases."""

    def __init__(self, y_true, p_var, p_grp, x_train,  x_test, model_object, model_type, model_name = 'auto', y_pred=None, y_prob=None, y_train=None, protected_features_cols=None, train_op_name="fit",
                 predict_op_name ="predict", feature_imp=None, sample_weight=None, pos_label=[[1]], neg_label=None):
        """
        Parameters
        ----------

        y_true : array of shape (n_samples,)
                Ground truth target values.

        y_pred : array of shape (n_samples,)
                Predicted targets as returned by classifier.

        y_prob : array of shape (n_samples,), default=None
                Predicted probabilities as returned by classifier.

        y_train : array of shape (m_samples,)
                Ground truth for training data.


        p_var : list of size=k
                List of privileged groups within the protected variables.

        p_grp : dictionary of lists, default = "auto"
                List of privileged groups within the protected variables.

        x_train : pandas Dataframe of shape (m_samples, n_features) or string
                Training dataset. m_samples refers to number of rows in the training dataset. The string refers to the dataset path acceptable by the model (e.g. HDFS URI).
        
        protected_features_cols: pandas Dataframe of shape (n_samples, k), default=None:
                This variable will be used for masking. If not provided, x_test will be used and x_test must be a pandas dataframe.

        x_test : pandas Dataframe or array of shape (n_samples, n_features) or string
                Testing dataset. The string refers to the dataset path acceptable by the model (e.g. HDFS URI).

        model_object : Object
                A blank model object used in the feature importance section for training and prediction.

        model_type : string
                The type of model to be evaluated in the feature importance section

                Customer Marketing: "uplift" or "propensity" or "reject"
                Credit Scoring: "default"

        Instance Attributes
        --------------------
        model_name : string, default="auto"
                Used to name the model artifact json file in compile function.

        y_pred : array of shape (n_samples,), default=None
                Predicted targets as returned by classifier.

        y_prob : array of shape (n_samples,), default=None
                Predicted probabilities as returned by classifier.

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

        _input_validation_lookup : dictionary
                Contains the attribute and its correct data type for every argument passed by user.

        err : object
                VeritasError object to save errors

        pos_label2 : array, default=None
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
        self.model_name = model_name[0:20]
        self.model_type = model_type
        if model_name == 'auto':
            self.model_name = model_type   
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.pos_label2 = None

        self.err = VeritasError()

        ## Define the ranges 

        
        #check_y_label = None

        check_y_label = list(set(y_true))
        check_p_grp ={}
        NoneType = type(None)
        check_model_type = []
        for usecase in Fairness.__subclasses__():
        #     print(usecase)
            model_type_to_metric_lookup = (getattr(usecase, "_model_type_to_metric_lookup"))
        #     print(list(model_type_to_metric_lookup.keys()))
            check_model_type = check_model_type + list(model_type_to_metric_lookup.keys())
        
        # Dictionary for expected data type
        # if value range is a tuple, will be condsidered as a numerical range
        # if value range is a list/set of str, will be considered as a collection of available values
        self._input_validation_lookup = {
        "y_true": [(list, np.ndarray, pd.Series), None],
        "y_train": [(NoneType, list, np.ndarray, pd.Series), check_y_label], ## y_pred unique values should be subset of y_true unique values
        "y_pred": [(NoneType, list, np.ndarray, pd.Series), check_y_label], ## y_pred unique values should be subset of y_true unique values
        "y_prob": [(NoneType, list, np.ndarray, pd.Series), (-0.01,1.01)],
        "p_var":  [(list,), str],
        "protected_features_cols": [(NoneType, pd.DataFrame), self.p_var],
        "p_grp":  [(dict,), check_p_grp], ## do a subset check. for each key in p_grp should be inside check_p_grp
        "x_train": [(pd.DataFrame, str), None],
        "x_test":  [(pd.DataFrame, str), None],
        "train_op_name": [(str,), None],
        "predict_op_name": [(str,), None],
        "feature_imp":     [(NoneType, pd.DataFrame), (np.dtype('O'), np.dtype('float64'))], ### use pandas.Dataframe.dtypes to check if the 1st columns is string (object) and 2nd column is numeric (float64)
        "sample_weight":   [(NoneType, list, np.ndarray), (0, np.inf)], ## should be > 0
        "model_name":      [(str,), None],
         "model_type":      [(str,), check_model_type],
        "pos_label":       [(list,), check_y_label], ## should be subset of y_true unique values
        "neg_label":       [(NoneType, list), check_y_label] ## should be subset of y_true unique values
        }
        
        # status, message = check_datatype(self)
        # #print(status)
        # #print(message)
        # assert status, message
        check_datatype(self)
        

        # Dictionary for input data
        if protected_features_cols is None and type(self.x_test) == pd.DataFrame:
            self.protected_features_cols = self.x_test[p_var]
        for var in self.p_grp.keys():
            check_p_grp[var] = self.protected_features_cols[var].unique()


        #converts to np array
        self.y_true = np.array(self.y_true)
        if self.y_pred is not None :
            self.y_pred = np.array(self.y_pred)
        if self.y_prob is not None :            
            self.y_prob = np.array(self.y_prob)
        if self.y_train is not None :
            self.y_train = np.array(self.y_train)
        if self.sample_weight is not None :
            self.sample_weight = np.array(self.sample_weight)

        # status, message = self.check_data_consistency()
        # #print(status)
        # #print(message)
        # assert status, message
        # self.check_data_consistency()
        # err_list = self.check_data_consistency()
        # if type(err_list) != str:
        #     self.err.push(err_list[0][0], var_name=err_list[0][1], given=err_list[0][2], expected=err_list[0][3])
        # else:
        #     print(err_list)
        # self.err.pop()


        #### in check_values, run a for loop to check if the range is not None, and check values.
        ###  in check_values, handle generic data types (dict, list, etc.)
        check_value(self)
        # status, message  = check_value(self)
        # #print(status)
        # #print(message)
        # assert status, message

        self.check_data_consistency()

        # status, message = self.check_data_consistency()
        # #print(status)
        # #print(message)
        # assert status, message

        #if user provide neg_label, need to 
        #label encoding to transfrom y_ture
        if len(self.y_true.shape) == 1 and self.y_true.dtype.kind in ['i','O','S']:
            self.y_true, self.pos_label2 = check_label(self.y_true, self.pos_label, self.neg_label)
        if self.y_pred is not None and len(self.y_pred.shape) == 1 and self.y_pred.dtype.kind in ['i','O','S']:
            self.y_pred, self.pos_label2 = check_label(self.y_pred, self.pos_label, self.neg_label)

        # if err_list != []:
        #     self.err.push(err_list[0][0], var_name_a=err_list[0][1], some_string=err_list[0][2], value=err_list[0][3])
        # else:
        #     print(err_list)
        # self.err.pop()

    def check_data_consistency(self):
        """
        Check rows and columns are of consistent size across the various datasets and the number & match of the unique values.

        Returns:
        ---------------
        err_ : string or list
                If err_ is an empty list, it will be a string that indicates no error was detected. Otherwise, it would be a list of errors

        """
        # this method only check the size
        err_ = []
        # check_status = True
        # errMsg = "data consistency error"
        successMsg = "data consistency check completed without issue"
        # errMsgFormat = "\n    {}: given {} size {}, expected {}"

        test_row_count = self.y_true.shape[0] #from y_true (ground truth for test dataset)pred on x_test, result in y_pred compare with y_true

        # feature_imp rowcount <= 20
        if self.feature_imp is not None:
            feature_imp_cols = len(self.feature_imp.columns)
            feature_imp_rows = len(self.feature_imp.index)
            if feature_imp_cols != 2:
                # errMsg += errMsgFormat.format("feature_imp", "column", str(feature_imp_cols), "2")
                err_.append(['length_error', "feature_imp column", str(feature_imp_cols), "2"])
                # check_status = check_status and False
            if feature_imp_rows > 20:
                # errMsg += errMsgFormat.format("feature_imp", "row", str(feature_imp_rows), "<=20")
                err_.append(['length_error', "feature_imp row", str(feature_imp_rows), "<=20"])
                # check_status = check_status and False

        # check protected_features_cols
        # check cols of protected_features_cols is same as p_var
        pro_f_cols_rows = len(self.protected_features_cols.index)
        pro_f_cols_cols = len(self.protected_features_cols.columns)
        if pro_f_cols_rows != test_row_count:
            # errMsg += errMsgFormat.format("protected_features_cols","row",str(pro_f_cols_rows),str(test_row_count))
            err_.append(['length_error', "protected_features_cols row", str(pro_f_cols_rows), str(test_row_count)])
            # check_status = check_status and False
        if pro_f_cols_cols != len(self.p_var):
            # errMsg += errMsgFormat.format("p_var","array",str(len(self.p_var)),str(pro_f_cols_cols))
            err_.append(['length_error', "p_var array", str(len(self.p_var)), str(pro_f_cols_cols)])
            # check_status = check_status and False


        # check x_train
        if type(self.x_train) == pd.DataFrame:
            train_row_count = self.x_train.shape[0]
            x_train_rows = len(self.x_train.index)
            if x_train_rows != train_row_count:
                # errMsg += errMsgFormat.format("x_train","row", str(x_train_rows), str(train_row_count))
                err_.append(['length_error', "x_train row", str(x_train_rows), str(train_row_count)])
                # check_status = check_status and False

        #check train datasets if y_train is not None
        if self.y_train is not None:
            train_row_count = self.y_train.shape[0] #from y_train  (ground truth for train dataset)
            # check x_train
            if type(self.x_train) == pd.DataFrame:
                x_train_rows = len(self.x_train.index)
                if x_train_rows != train_row_count:
                    err_.append(['length_error', "x_train row", str(x_train_rows), str(train_row_count)])
                    # errMsg += errMsgFormat.format("x_train","row", str(x_train_rows), str(train_row_count))
                    # check_status = check_status and False


        #y_prob should be float
        if self.y_prob is not None and self.y_prob.dtype.kind != 'f':
            self.err.push('type_error', var_name="y_prob", given= "not type float64", expected="float64", function_name="check_data_consistency")

        # if both x_test and x_train are df, check they have same no. of columns
        if type(self.x_train) == pd.DataFrame and type(self.x_test) == pd.DataFrame:
            x_train_cols = len(self.x_train.columns)
            x_test_cols = len(self.x_test.columns)
            if x_train_cols != x_test_cols:
                # errMsg += errMsgFormat.format("x_train","column", str(x_train_cols), str(x_test_cols))
                err_.append(['length_error', "x_train column", str(x_train_cols), str(x_test_cols)])
                # check_status = check_status and False

        #check pos_label size and neg_label size

        try:
            for usecase in Fairness.__subclasses__():
                model_type_to_metric_lookup = (getattr(usecase, "_model_type_to_metric_lookup"))
                if self.model_type in model_type_to_metric_lookup.keys():
                        #assumption: model_type is unique across all use cases
                    label_size = (model_type_to_metric_lookup.get(self.model_type)[1])/2
                    break
        except:
            pass

        # if label size requirement is -1/2, no need to check

        try:
            if label_size > 0:
                if self.neg_label is not None:
                    neg_label_size = len(self.neg_label)
                    if neg_label_size != label_size:
                        # errMsg += errMsgFormat.format("neg_label", "", str(neg_label_size), str(label_size))
                        err_.append(['length_error', "x_train column", str(x_train_cols), str(x_test_cols)])
                        # check_status = check_status and False
                    pos_label_size = len(self.pos_label)
                    if pos_label_size != label_size:
                        # errMsg += errMsgFormat.format("pos_label", "", str(pos_label_size), str(label_size))
                        err_.append(['length_error', "pos_label", str(pos_label_size), str(label_size)])
                        # check_status = check_status and False
        except:
            pass

        #y_pred and y_prob should not be both none
        if self.y_pred is None and self.y_prob is None:
            # errMsg += errMsgFormat.format("y_pred and y_prob", "None for both", "not both are None")
            err_.append(['length_error', "y_pred and y_prob", "None for both", "not both are None"])
            # check_status = check_status and False

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
                    try:
                        if var_name =="y_prob" and check_order[i] == "column" and label_size == 2:
                            expected_size = 4
                    except:
                        pass
                        
                    if given_size != expected_size:
                        # errMsg += errMsgFormat.format(var_name, check_order[i], str(given_size), str(expected_size))
                        err_.append(['length_error', var_name + " " + check_order[i], str(given_size), str(expected_size)])
                        # check_status = check_status and False

        if err_ == []:
            err_ = successMsg
            return err_
        else:
            self.err.push(err_[0][0], var_name=err_[0][1], given=err_[0][2], expected=err_[0][3],
                          function_name="check_data_consistency")
            self.err.pop()


        # msg = successMsg if check_status else errMsg
        # return check_status, msg
    
    def clone(self,  y_true, model_object, y_pred=None, y_prob=None, y_train=None, train_op_name="fit",
                 predict_op_name ="predict", feature_imp=None, sample_weight=None, pos_label=[[1]], neg_label=None):

        """

        Parameters
        ---------------
        y_true : array of shape (n_samples,)
                Ground truth target values.

        model_object : Object
                A blank model object used in the feature importance section for training and prediction.

        model_name : string, default="auto"
                Used to name the model artifact json file in compile function.

        y_pred : array of shape (n_samples,), default=None
                Predicted targets as returned by classifier.

        y_prob : array of shape (n_samples,), default=None
                Predicted probabilities as returned by classifier.

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

