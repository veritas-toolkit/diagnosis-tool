import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix
from copy import deepcopy
import multiprocessing
import math
from ..config.constants import Constants
from .errors import *
from pathlib import Path

def check_datatype(obj_in):
    """
    Checks whether a particular input is of the correct datatype.

    Parameters
    ----------------
    obj_in : object
            Object that needs to be checked

    Returns:
    ---------------
    successMsg : string
            If there are no errors, a success message will be returned   
    """
    err = VeritasError()
    successMsg = "data type check completed without issue"
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
        if exp_type is None:
            continue
        #get the variable
        try:
            var = getattr(obj_in, var_name)
        except:
            if(NoneType not in exp_type):
                err_.append(['type_error', str(var_name), "None", str(exp_type)])
                continue
        #get variable type
        cur_type = type(var)

        #check datatype
        if cur_type not in exp_type:
            err_.append(['type_error', str(var_name), str(cur_type), str(exp_type)])

        if var_name == "p_grp":
            if cur_type in exp_type :
                for i in getattr(obj_in, var_name).values():
                    if type(i) != list:
                        err_.append(['type_error', "p_grp values", str(type(i)), "list"])

    if err_ == []:
        return successMsg
    else:
        for i in range(len(err_)):
            err.push(err_[i][0], var_name=err_[i][1], given=err_[i][2], expected=err_[i][3], function_name="check_datatype")
        err.pop()

def check_value(obj_in):
    """
    Checks if certain values are present in the obj_in. E.g. Check if the the performance and fairness metrics provided by user (if any) are supported

    Parameters
    ----------------
    obj_in : object
            Object that needs to be checked

    Returns:
    ---------------
    successMsg : string
            If there are no errors, a success message will be returned   
    """
    err = VeritasError()
    err_ = []
    input_validation_lookup = obj_in._input_validation_lookup
    var_names = input_validation_lookup.keys()
    successMsg = "data value check completed without issue"
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

        #only perform check_value for range provided variables
        if(len(input_validation_lookup.get(var_name))==2):
            var_range = input_validation_lookup.get(var_name)[1]
            var_range_type = type(var_range)
        else:
            continue
        
        if var_value is None or var_range is None:
            continue

        elif var_value_type in str_type and var_range_type in collection_types:
            if not var_value in var_range:
                err_.append(['value_error', var_name, str(var_value), str(var_range)])
        
        # eg. y_pred, pos_label, neg_label
        elif var_value_type in collection_types and var_range_type in collection_types:
            var_value = set(np.array(var_value).ravel())
            var_range = set(var_range)
            if not var_value.issubset(var_range):
                err_.append(['value_error', var_name, str(sorted(list(var_value))), str(sorted(list(var_range)))])
        
        # eg. check p_var
        elif var_value_type in collection_types and var_range_type == type:
            for i in var_value:
                if type(i) != var_range:
                    err_.append(['value_error', var_name, str(type(i)), str(str)])
        
        # eg. protected_features_cols
        elif var_value_type == pd.DataFrame and var_range_type in range_types:
            column_names = set(var_value.columns.values)
            if not column_names.issubset(set(var_range)):
                err_.append(['column_value_error', var_name, str(sorted(list(var_range))), str(sorted(list(column_names)))])
        
        # eg check y_prob
        elif var_value_type in numeric_list_types and var_range_type in numeric_range_types:
            # need to perfrom check on whether the dimension array
            min_value = var_value.min()
            max_value = var_value.max()
            if min_value < var_range[0] or max_value > var_range[1]:
                err_.append(['value_error', var_name, "range [" + str(min_value) + " : " + str(max_value) + "] ",
                             str(var_range)])
        
        # eg check fair_neutral_tolerance
        elif var_value_type in numeric_types and var_range_type in numeric_range_types:
            if var_value < var_range[0] or var_value > var_range[1]:
                err_.append(['value_error', var_name, str(var_value), str(var_range)])
        
        # eg check feature_imp
        elif var_value_type == pd.DataFrame and var_range_type in numeric_range_types:
            var_value_types = var_value.dtypes
            if len(var_value_types) != len(var_range):
                err_.append(['length_error', var_name, len(var_value_types), len(var_range)])
            else:
                for i, tp in zip(range(len(var_value_types)), var_value_types):
                    if tp != var_range[i]:
                        err_.append(['column_value_error', var_name, str(tp), str(var_range[i])])
        
        # eg check p_grp
        elif var_value_type == dict and var_range_type == dict:
            keyset1 = set(var_value.keys())
            keyset2 = set(var_range.keys())
            if keyset1 != keyset2:
                err_.append(['value_error', var_name, str(sorted(list(keyset1))), str(sorted(list(keyset2)))])
            else:
                for key in keyset1:
                    i_var = convert_to_set(var_value.get(key))
                    i_range = convert_to_set(var_range.get(key))
                    if not i_var.issubset(i_range):
                        err_.append(['value_error', var_name + " " + key, str(sorted(list(i_var))), str(sorted(list(i_range)))])
        
        else:
            err_.append(['value_error', var_name, "a range of " + str(var_range), "a range for " + str(var_value_type)])

    if err_ == []:
        return successMsg
    else:
        for i in range(len(err_)):
            err.push(err_[i][0], var_name=err_[i][1], given=err_[i][2], expected=err_[i][3], function_name="check_value")
        err.pop()

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

def check_label(y, pos_label, neg_label=None):
    """
    Creates copy of y_true as y_true_bin and convert favourable labels to 1 and unfavourable to 0 for non-uplift models.
    Overwrite y_pred with the conversion.
    Checks if pos_labels are inside y

    Parameters
    -----------
    y : array of shape (n_samples,)
            Ground truth target values.

    pos_label : array
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
    # uplift model
    # 0, 1 => control (others, rejected/responded)
    # 2, 3 => treatment (others, rejected/responded)
    err = VeritasError()
    err_= []

    if neg_label is not None and len(neg_label) == 2:
        y_bin = y
        n=0

        row = y_bin == pos_label[0]  
        indices_pos_0 = [i for i, x in enumerate(y_bin) if x == pos_label[0][0]]
        n += np.sum(row)

        row = y_bin == pos_label[1]  
        indices_pos_1 = [i for i, x in enumerate(y_bin) if x == pos_label[1][0]]
        n += np.sum(row)

        row = y_bin == neg_label[0]  
        indices_neg_0 = [i for i, x in enumerate(y_bin) if x == neg_label[0][0]]
        n += np.sum(row)

        row = y_bin == neg_label[1]  
        indices_neg_1 = [i for i, x in enumerate(y_bin) if x == neg_label[1][0]]
        n += np.sum(row)     

        for i in indices_pos_0:
            y_bin[i] = "TR"
        for i in indices_pos_1:
            y_bin[i] = "CR"
        for i in indices_neg_0:
            y_bin[i] = "TN"
        for i in indices_neg_1:
            y_bin[i] = "CN"

        if n != len(y_bin):
            err_.append(['conflict_error', "pos_label, neg_label", "inconsistent values", pos_label + neg_label])
            for i in range(len(err_)):
                err.push(err_[i][0], var_name_a=err_[i][1], some_string=err_[i][2], value=err_[i][3],
                        function_name="check_label")
        pos_label2 = [['TR'],['CR']]
    
    else:
        y_bin = y
        row = np.isin(y_bin, pos_label[0])
        if sum(row) == len(y_bin) :
            err_.append(['value_error', "pos_label", pos_label[0], "not all y_true labels"])
        elif sum(row) == 0 :
            err_.append(['value_error', "pos_label", pos_label[0], set(y_bin)])            
        for i in range(len(err_)):
            err.push(err_[i][0], var_name=err_[i][1], given=err_[i][2], expected=err_[i][3],
                    function_name="check_label")

        y_bin[row] = 1 
        y_bin[~row] = 0
        pos_label2 = [[1]]
    
    if y_bin.dtype.kind in ['i']:
        y_bin  = y_bin.astype(np.int8)

    err.pop()

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
            Number of currently active threads of a job

    Returns
    -------------
    n_threads : int
            Number of currently active threads of a job
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
    from ..fairness import CreditScoring
    from ..model import ModelContainer
    import pickle
    #Load Credit Scoring Test Data
    PATH = Path(__file__).parent.parent.joinpath('resources', 'data')
    file = PATH/"credit_score_dict.pickle"
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
    model_object = cs["model"]
    model_name = "credit scoring"
    model_type = "credit"
    y_prob = cs["y_prob"]

    #rejection inference
    num_applicants = {'SEX': [3500, 5000], 'MARRIAGE':[3500, 5000]}
    base_default_rate = {'SEX': [0.10,0.05], 'MARRIAGE':[0.10,0.05]}
    
    container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,model_name =  model_name, y_pred= y_pred, y_prob= y_prob)
    cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "balanced_acc", fair_metric_name = "equal_opportunity")
    cre_sco_obj.k = 1
    cre_sco_obj.evaluate(output = False)
    result = cre_sco_obj.perf_metric_obj.result, cre_sco_obj.fair_metric_obj.result
    file = PATH/"credit_score_sample.pickle"
    input_file = open(file, "rb")
    sample = pickle.load(input_file)
    if result[0]['perf_metric_values'] == sample[0]['perf_metric_values'] and result[1] == sample[1]:
        print("Evaluation of credit scoring performed normally")
    else:
        print('The test results are abnormal')
