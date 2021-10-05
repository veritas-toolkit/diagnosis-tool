import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from copy import deepcopy
import multiprocessing
import math
from .config.constants import Constants
from .ErrorCatcher import VeritasError

def check_datatype(obj_in):
    """
    Checks whether a particular input is of the correct datatype.

    Parameters
    ----------------
    obj_in : object
            Object that needs to be checked

    Returns:
    ---------------
    err_ : string or list
            If err_ is an empty list, it will be a string that indicates no error was detected. Otherwise, it would be a list of errors

    """

    err = VeritasError()
    # errMsg = "data type error"
    successMsg = "data type check completed without issue"
    # errMsgFormat = "\n    {}: given {}, expected {}"
    # check_status = True
    NoneType = type(None)
    err_ = []
    input_validation_lookup = obj_in._input_validation_lookup

    #loop through each variable specified inside _input_validation_lookup
    var_names = input_validation_lookup.keys()
    for var_name in var_names:
        exp_type = input_validation_lookup.get(var_name)[0]
        #convert exp_type into iterable list
        if type(exp_type) == tuple:
            exp_type = list(exp_type)
        else:
            emptylist = list()
            exp_type = emptylist.append(exp_type)
        if exp_type == None:
            continue
        #get the variable
        try:
            var = getattr(obj_in, var_name)
        except:
            if(NoneType not in exp_type):
                # errMsg += errMsgFormat.format(str(var_name),"None", str(exp_type))
                err_.append(['type_error', str(var_name), "None", str(exp_type)])
                # check_status = check_status and False
                continue
        #get variable type
        cur_type = type(var)

        #check datatype
        if cur_type not in exp_type:
            # errMsg += errMsgFormat.format(str(var_name),str(cur_type), str(exp_type))
            err_.append(['type_error', str(var_name), str(cur_type), str(exp_type)])
            # check_status = check_status and False
        #to check callable functions
        if var_name in ["train_op_name", "predict_op_name"]:
            try:
                callable(getattr(obj_in.model_object, var))
            except AttributeError:
                # errMsg += errMsgFormat.format("'"+var_name+"'"," non-callable " + "'"+var_name+"'", "callable"+str(var_name)) #check this line
                err_.append(['type_error', var_name, " non-callable ", "callable"])
                # check_status = check_status and False

    if err_ == []:
        err_ = successMsg
        return err_
    else:
        err.push(err_[0][0], var_name=err_[0][1], given=err_[0][2], expected=err_[0][3], function_name="check_datatype")
        err.pop()

    # return err_
    # msg = successMsg if check_status else errMsg
    # return check_status, msg

def check_value(obj_in):
    """
    Checks if certain values are present in the obj_in. E.g. Check if the the performance and fairness metrics provided by user (if any) are supported

    Parameters
    ----------------
    obj_in : object
            Object that needs to be checked

    Returns:
    ---------------
    err_ : string or list
            If err_ is an empty list, it will be a string that indicates no error was detected. Otherwise, it would be a list of errors
    """

    err = VeritasError()
    err_ = []
    input_validation_lookup = obj_in._input_validation_lookup
    var_names = input_validation_lookup.keys()
    # errMsg ="data value error"
    successMsg = "data value check completed without issue"
    # errMsgFormat = "\n    {}: given {}, expected {}"
    # check_status = True
    numeric_types = [int, float]
    numeric_list_types = [list, np.array, np.ndarray]
    numeric_range_types = [tuple,]
    range_types = [list,]
    str_type = [str,]
    collection_types = [list, set, np.ndarray]

    for var_name in var_names:
        var_value = getattr(obj_in, var_name)
        var_value_type = type(var_value)
        var_range = None
        #print("=============checking==========  " +var_name)
        #only perform check_value for range provided variables
        if(len(input_validation_lookup.get(var_name))==2):
            var_range = input_validation_lookup.get(var_name)[1]
            var_range_type = type(var_range)
        else:
            continue
        if var_value is None or var_range is None:
            continue
        elif var_value_type in numeric_types and var_range_type in range_types:
            if len(var_range) != 2 or type(var_range[0]) not in numeric_types or type(var_range[1]) not in numeric_types:
                # errMsg += errMsgFormat.format(var_name, str(var_range), "a range for "+str(var_value_type))
                err_.append(['value_error', var_name, str(var_range), "a range for " + str(var_value_type)])
                # check_status = check_status and False
            if not(var_value > var_range[0] and var_value < var_range[1]):
                # errMsg += errMsgFormat.format(var_name, str(var_value), str(var_range))
                err_.append(['value_error', var_name, str(var_value), str(var_range)])
                # check_status = check_status and False
        elif var_value_type in str_type and var_range_type in collection_types:
            if not var_value in var_range:
                # errMsg += errMsgFormat.format(var_name, str(var_value), str(var_range))
                err_.append(['value_error', var_name, str(var_value), str(var_range)])
                # check_status = check_status and False
        # eg. y_pred, pos_label, neg_label
        elif var_value_type in collection_types and var_range_type in collection_types:
            var_value = set(np.array(var_value).ravel())
            var_range = set(var_range)
            if not var_value.issubset(var_range):
                # errMsg += errMsgFormat.format(var_name, str(var_value), str(var_range))
                err_.append(['value_error', var_name, str(var_value), str(var_range)])
                # check_status = check_status and False
        # eg. check p_var
        elif var_value_type in collection_types and var_range_type == type:
            for i in var_value:
                if type(i) != var_range:
                    # errMsg += errMsgFormat.format(var_name, str(type(i)), str(str))
                    err_.append(['value_error', var_name, str(type(i)), str(str)])
                    # check_status = check_status and False
        # eg. protected_features_cols
        elif var_value_type == pd.DataFrame and var_range_type in range_types:
            column_names = set(var_value.columns.values)
            if not column_names.issubset(set(var_range)):
                # errMsg += errMsgFormat.format(var_name, str(column_names), str(var_range))
                err_.append(['column_value_error', var_name, str(var_range), str(column_names)])
                # check_status = check_status and False
        # eg check y_prob
        elif var_value_type in numeric_list_types and var_range_type in numeric_range_types:
            # need to perfrom check on whether the dimension array
            min_value = var_value.min()
            max_value = var_value.max()
            if min_value < var_range[0] or max_value > var_range[1]:
                # errMsg += errMsgFormat.format(var_name, "range [" +str(min_value) +" : " + str(max_value) +"] ", str(var_range))
                err_.append(['value_error', var_name, "range [" + str(min_value) + " : " + str(max_value) + "] ",
                             str(var_range)])
                # check_status = check_status and False
        # eg check fair_neutral_tolerance
        elif var_value_type in numeric_types and var_range_type in numeric_range_types:
            if var_value < var_range[0] or var_value > var_range[1]:
                # errMsg += errMsgFormat.format(var_name, "range [" +str(min_value) +" : " + str(max_value) +"] ", str(var_range))
                err_.append(['value_error', var_name, "range [" + str(min_value) + " : " + str(max_value) + "] ",
                             str(var_range)])
                # check_status = check_status and False
        # eg check feature_imp
        elif var_value_type == pd.DataFrame and var_range_type in numeric_range_types:
            var_value_types = var_value.dtypes
            for i, tp in zip(range(len(var_value_types)), var_value_types):
                if tp != var_range[i]:
                    # errMsg += errMsgFormat.format(var_name + "column " +str(i), str(tp), str(var_range[i]))
                    err_.append(['column_value_error', var_name, str(var_range[i])], str(i))
                    # check_status = check_status and False
        # eg check p_grp
        elif var_value_type == dict and var_range_type == dict:
            keyset1 = set(var_value.keys())
            keyset2 = set(var_range.keys())
            if keyset1 != keyset2:
                # errMsg += errMsgFormat.format(var_name, str(keyset1), str(keyset2))
                err_.append(['value_error', var_name, str(keyset1), str(keyset2)])
                # check_status = check_status and False
            else:
                for key in keyset1:
                    i_var = convert_to_set(var_value.get(key))
                    i_range = convert_to_set(var_range.get(key))
                    if not i_var.issubset(i_range):
                        # errMsg += errMsgFormat.format(var_name + " " +key, str(i_var), str(i_range) )
                        err_.append(['value_error', var_name + " " + key, str(i_var), str(i_range)])
                        # check_status = check_status and False
        # eg check base_default_rate and num_rejected_applicant
        elif var_value_type == dict and var_range_type == list:
            keyset1 = set(var_value.keys())
            keyset2 = set(var_range_type)
            if not keyset1.issubset(keyset2):
                # errMsg += errMsgFormat.format(var_name, str(keyset1), str(keyset2))
                err_.append(['value_error', var_name, str(keyset1), str(keyset2)])
                # check_status = check_status and False
        # eg check feature_imp
        elif var_value_type == pd.DataFrame and var_range_type in collection_types:
            for i in range(len(var_value.columns)):
                i_var_type = var_value.iloc[:, i].dtypes
                i_var_range = var_range[i]
                if(i_var_type != i_var_range):
                    # errMsg += errMsgFormat.format(var_name +" "+ var_value.columns.values[i], i_var_type, i_var_range)
                    err_.append(['column_value_error', var_name, i_var_range, var_value.columns.values[i]])
                    # check_status = check_status and False
        else:
            # errMsg += errMsgFormat.format(var_name, "a range of "+str(var_range), "a range for "+str(var_value_type))
            err_.append(['value_error', var_name, "a range of " + str(var_range), "a range for " + str(var_value_type)])
            # check_status = check_status and False

    if err_ == []:
        err_ = successMsg
        return err_
    else:
        err.push(err_[0][0], var_name=err_[0][1], given=err_[0][2], expected=err_[0][3], function_name="check_value")
        err.pop()

    # return err_

def convert_to_set(var):
    """
    Converts certain types of variable into set

    Parameters
    ----------
    var : 

    Returns 
    ----------
    result : 
    """
    result = set()
    if type(var) in [int, float, str]:
        result = {var,}
    elif type(var) == set:
        result = var
    elif type(var) in [list, tuple]:
        result = set(var)
    else:
        result = var
    return result

def bootstrap_conf_int(metric_obj, metric_name, k=50):
    """
    Calculates the confidence interval for any given metric

    Parameters
    ----------------
    metric_obj : object
            A single initialized Fairness use case object (CreditScoring, CustomerMarketing, etc.)

    metric_name : str
            The performance or fairness metric name.

    k : int, default=50
            The number of data points computed to obtain the confidence interval

    Returns:
    ---------------
    np.mean(out), 2*np.std(out) : tuple of floats
            Mean and confidence interval values
    """
    n = len(metric_obj.y_true[0])
    out = []
    np.random.seed(123)
    for i in range(k):
        idx = np.random.choice(n, n, replace=True)
        out.append(metric_obj.translate_metric(metric_name, idx=idx)[0])
    out = np.array(out)
    return np.mean(out), 2*np.std(out)

    
    
def format_uncertainty(mean_val, conf_int):
    """
    Formats uncertainty

    Parameters
    ------------
    mean_val: float
            Mean value

    conf_int: float
            Confidence interval value

    Returns
    ------------
    Mean and confidence interval in standardised number of decimal places
    """
    # return f"{mean_val:.3f} +/- {conf_int:.3f}"
    return "{:.{decimal_pts}} +/- {:.{decimal_pts}}".format(mean_val, conf_int, decimal_pts=Constants().decimals)


def check_label(y, pos_label, neg_label=None):
    """
    Creates copy of y_true as y_true_bin and convert favourable labels to 1 and unfavourable to 0 for non-uplift models.
    Overwrite y_pred with the conversion.

    Parameters
    -----------
    y : array of shape (n_samples,)
            Ground truth target values.

    pos_label : array, default = [[1]]
            Label values which are considered favorable.
            For all model types except uplift, converts the favourable labels to 1 and others to 0.
            For uplift, user is to provide 2 label names e.g. [["a"], ["b"]] in fav label. The first will be mapped to treatment responded (TR) & second to control responded (CR).

    neg_label : array, default=None
            Label values which are considered unfavorable.
            neg_label will only be used in uplift models.
            For uplift, user is to provide 2 label names e.g. [["c"], ["d"]] in unfav label. The first will be mapped to treatment rejected (TR) & second to control rejected (CR).


    Returns
    -----------------
    y_bin : list
            Encoded labels.

    pos_label2 : array
            Label values which are considered favorable.

    """
    ## check if fav_label is inside unique values in y_true
    ## uplift model
    ## 0, 1 => control (others, rejected/responded)
    ## 2, 3 => treatment (others, rejected/responded)
    err = VeritasError()
    err_= []
    if neg_label is not None and len(neg_label) == 2:
        y_bin = y
        n=0
        row = y_bin == pos_label[0]  
        y_bin[row] = 'TR' 
        n += np.sum(row)
        row = y_bin == pos_label[1]  
        y_bin[row] = 'CR' 
        n += np.sum(row)
        row = y_bin == neg_label[0]  
        y_bin[row] = 'TN' 
        n += np.sum(row)
        row = y_bin == neg_label[1]  
        y_bin[row] = 'CN' 
        n += np.sum(row)        
        if n != len(y_bin):
            # raise ValueError('y dataset contains labels other than provided. Please provide valid pos_label and neg_label.')
            err_.append(['conflict_error', "y dataset", "inconsistent values", pos_label + neg_label])
            err.push(err_[0][0], var_name_a=err_[0][1], some_string=err_[0][2], value=err_[0][3],
                     function_name="check_label")
            err.pop()
        pos_label2 = [['TR'],['CR']]
    else:
        y_bin = y
        row = y_bin == pos_label[0]  
        y_bin[row] = 1 
        y_bin[~row] = 0
        pos_label2 = [[1]]
    
    if y_bin.dtype.kind == 'i':
        y_bin  = y_bin.astype(np.int8)


    return y_bin, pos_label2


def get_cpu_count():
    """
    Get the number of CPUs of machine that toolkit is running on.

    Returns
    --------
    CPU count
    """
    return multiprocessing.cpu_count()

def check_multiprocessing(n_threads):
    """
    Determine the number of threads/processes for parallelization.
    0 means auto, else the number is capped by CPU count as well

    Parameters
    -------------
    n_threads : int
            number of currently active threads of a job


    Returns
    -------------
    n_threads : int
            number of currently active threads of a job
    """
    
    if n_threads == 1:
        n_threads = 1
        
    elif n_threads == 0:
        n_threads = math.floor(get_cpu_count()/2)

    elif n_threads > 1 :
        n_threads = min(math.floor(get_cpu_count()/2),n_threads) 
    
    else :
        n_threads = 1
 
    return n_threads

def test_function_cs():
    #Load Credit Scoring Test Data
    file = r".\veritas\resources\data\credit_score_dict.pickle"
    input_file = open(file, "rb")
    cs = pickle.load(input_file)

    #Reduce into two classes
    cs["X_train"]['MARRIAGE'] = cs["X_train"]['MARRIAGE'].replace([0, 3],1)
    cs["X_test"]['MARRIAGE'] = cs["X_test"]['MARRIAGE'].replace([0, 3],1)
    #Model Contariner Parameters
    y_true = np.array(cs["y_test"])
    y_pred = np.array(cs["y_pred"])
    y_train = np.array(cs["y_train"])
    p_var = ['SEX', 'MARRIAGE']
    p_grp = {'SEX': [1], 'MARRIAGE':[1]}
    x_train = cs["X_train"]
    x_test = cs["X_test"]
    model_object = LogisticRegression(C=0.1)
    model_name = "credit scoring"
    model_type = "default"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
    base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}
    model_object = LRwrapper(model_object)
    
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "balanced_acc", fair_metric_name = "equal_opportunity")
    cre_sco_obj.k = 1
    cre_sco_obj.evaluate(output = False)
    result = cre_sco_obj.perf_metric_obj.result, cre_sco_obj.fair_metric_obj.result
    file = r".\veritas\resources\data\credit_score_sample.pickle"
    input_file = open(file, "rb")
    sample = pickle.load(input_file)
    if result[0]['perf_metric_values'] == sample[0]['perf_metric_values'] and result[1] == sample[1]:
        print("Evaluation of credit scoring performed normally")
    else:
        print('The test results are abnormal')
        
def test_function_cm():
#Load Phase 1-Customer Marketing Uplift Model Data, Results and Related Functions
    file_prop = r".\veritas\resources\data\mktg_uplift_acq_dict.pickle"
    file_rej = r".\veritas\resources\data\mktg_uplift_rej_dict.pickle"
    input_prop = open(file_prop, "rb")
    input_rej = open(file_rej, "rb")
    cm_prop = pickle.load(input_prop)
    cm_rej = pickle.load(input_rej)

    #Model Container Parameters
    #Rejection Model
    y_true_rej = cm_rej["y_test"]
    y_pred_rej = cm_rej["y_test"]
    y_train_rej = cm_rej["y_train"]
    p_var_rej = ['isforeign', 'isfemale']
    p_grp_rej = {'isforeign':[0], 'isfemale':[0]}
    x_train_rej = cm_rej["X_train"].drop(['ID'], axis = 1)
    x_test_rej = cm_rej["X_test"].drop(['ID'], axis = 1)
    model_object_rej = cm_rej['model']
    model_name_rej = "cm_rejection"
    model_type_rej = "uplift"
    y_prob_rej = cm_rej["y_prob"]
    # y_prob_rej = None
    data = {"FEATURE" :['income', 'noproducts', 'didrespond', 'age', 'isfemale',
           'isforeign'], 
            "VALUE":[0.3, 0.2, 0.15, 0.1, 0.05, 0.03]}
    feature_importance_prop = pd.DataFrame(data)

    #Propensity Model
    y_true_prop = cm_prop["y_test"]
    y_pred_prop = cm_prop["y_test"]
    y_train_prop = cm_prop["y_train"]
    p_var_prop = ['isforeign', 'isfemale']
    p_grp_prop = {'isforeign':[0], 'isfemale':[0]}
    x_train_prop = cm_prop["X_train"].drop(['ID'], axis = 1)
    x_test_prop = cm_prop["X_test"].drop(['ID'], axis = 1)
    model_object_prop = cm_prop['model']
    model_name_prop = "cm_propensity" 
    model_type_prop = "uplift"
    y_prob_prop = cm_prop["y_prob"]
    #y_prob_prop = None
    data = {"FEATURE" :['income', 'noproducts', 'didrespond', 'age', 'isfemale',
           'isforeign'], 
            "VALUE":[0.3, 0.2, 0.15, 0.1, 0.05, 0.03]}
    feature_importance_rej = pd.DataFrame(data)

    PROFIT_RESPOND = 190
    COST_TREATMENT =20
    
    container_rej = ModelContainer(y_true = y_true_rej, y_pred = y_true_rej, y_prob = y_prob_rej, y_train= y_train_rej, p_var = p_var_rej, p_grp = p_grp_rej, x_train = x_train_rej,  x_test = x_test_rej, model_object = model_object_rej,  model_name = model_name_rej, model_type = model_type_rej,  pos_label=[['TR'], ['CR']], neg_label=[['TN'], ['CN']], predict_op_name = "predict_proba", feature_imp = feature_importance_rej)
    container_prop = container_rej.clone(y_true = y_true_prop,  y_train = y_train_prop, model_object = model_object_prop, y_pred=None, y_prob=y_prob_prop, train_op_name="fit",
                 predict_op_name ="predict_proba", feature_imp=None, sample_weight=None, pos_label=[['TR'], ['CR']], neg_label=[['TN'], ['CN']])

    cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 0.2, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "expected_profit", revenue = PROFIT_RESPOND, treatment_cost =COST_TREATMENT)
    cm_uplift_obj.k = 1
    cm_uplift_obj.evaluate(output = False)
    result = cm_uplift_obj.perf_metric_obj.result, cm_uplift_obj.fair_metric_obj.result
    file = r".\veritas\resources\data\cm_uplift_sample.pickle"
    input_file = open(file, "rb")
    sample = pickle.load(input_file)
    if result[0]['perf_metric_values'] == sample[0]['perf_metric_values'] and result[1] == sample[1]:
        print("Evaluation of customer marketing performed normally")
    else:
        print('The test results are abnormal')
