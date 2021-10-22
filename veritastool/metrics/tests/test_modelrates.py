import pickle
import numpy as np
import pytest
import pandas as pd
from copy import deepcopy
from veritastool.metrics.modelrates import *
from veritastool.model.model_container import ModelContainer
from veritastool.fairness.customer_marketing import CustomerMarketing
import sys
sys.path.append("veritastool/examples/customer_marketing_example")
import selection, uplift, util

#Load Credit Scoring Test Data
file = "veritastool/examples/data/credit_score_dict.pickle"
input_file = open(file, "rb")
cs = pickle.load(input_file)

y_true = np.array(cs["y_test"])
y_prob = cs["y_prob"]


def test_ModelRateClassify_init():
    sample_weight = np.random.choice(10, 7500, replace=True)
    modelrate_obj = ModelRateClassify(y_true, y_prob, sample_weight = sample_weight)
    assert modelrate_obj.tpr([0.5])[0]  <= 0.7 and modelrate_obj.tpr([0.5])[0]  >= 0.6
    assert modelrate_obj.fpr([0.5])[0] <= 0.37 and modelrate_obj.fpr([0.5])[0] >= 0.30
    assert modelrate_obj.ppv([0.5])[0] <= 0.92 and modelrate_obj.ppv([0.5])[0] >= 0.82
    assert modelrate_obj.forr([0.5])[0] <= 2 and modelrate_obj.forr([0.5])[0] >= 1.75
    assert modelrate_obj.selection_rate([0.5])[0] <= 0.65 and modelrate_obj.selection_rate([0.5])[0] >= 0.5
    assert round(modelrate_obj.base_selection_rate,2) <= 0.79 and round(modelrate_obj.base_selection_rate,2) >= 0.77


def test_compute_rates():
    ModelRateClassify.compute_rates(y_true, y_prob, sample_weight = None)
    ths, tpr, fpr, ppv, forr, base_selection_rate, selection_rate = ModelRateClassify.compute_rates(y_true, y_prob, sample_weight = None)

    assert ths.shape == (2174,)
    assert tpr.shape == (2174,)
    assert tpr.mean() == 0.6417562335342573
    assert fpr.shape == (2174,)
    assert fpr.mean() == 0.40266439975312385
    assert ppv.shape == (2174,)
    assert ppv.mean() == 0.8627624081302739
    assert forr.shape == (2174,)
    assert forr.mean() == 9.395396004091237
    assert base_selection_rate == 0.7788
    assert selection_rate.shape == (2174,)
    assert selection_rate.mean() == 0.5888691199018706


def test_ModelRateUplift_init():
    #Load Phase 1-Customer Marketing Uplift Model Data, Results and Related Functions
    # file_prop = r"C:\Users\brian.zheng\OneDrive - Accenture\Desktop\Veritas\Development\veritas_v1\pickle_files\mktg_uplift_acq_dict.pickle"
    # file_rej = r"C:\Users\brian.zheng\OneDrive - Accenture\Desktop\Veritas\Development\veritas_v1\pickle_files\mktg_uplift_rej_dict.pickle"
    file_prop = "veritastool/examples/data/mktg_uplift_acq_dict.pickle"
    file_rej = "veritastool/examples/data/mktg_uplift_rej_dict.pickle"
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
    data = {"FEATURE" :['income', 'noproducts', 'didrespond', 'age', 'isfemale',
           'isforeign'], 
            "VALUE":[0.3, 0.2, 0.15, 0.1, 0.05, 0.03]}
    feature_importance_rej = pd.DataFrame(data)

    PROFIT_RESPOND = 190
    COST_TREATMENT =20

    container_rej = ModelContainer(y_true = y_true_rej, y_pred = y_pred_rej, y_prob = y_prob_rej, y_train= y_train_rej, p_var = p_var_rej, p_grp = p_grp_rej, x_train = x_train_rej,  x_test = x_test_rej, model_object = model_object_rej,  model_name = model_name_rej, model_type = model_type_rej,  pos_label=[['TR'], ['CR']], neg_label=[['TN'], ['CN']], predict_op_name = "predict_proba",  feature_imp = feature_importance_rej)

    container_prop = container_rej.clone(y_true = y_true_prop, y_pred = y_pred_prop, y_prob = y_prob_prop, y_train= y_train_prop,\
                                    model_object = model_object_prop,  pos_label=[['TR'], ['CR']], neg_label=[['TN'], ['CN']], \
                                    predict_op_name = "predict_proba", feature_imp = feature_importance_prop)

    cm_uplift_obj = CustomerMarketing(model_params = [container_rej, container_prop], fair_threshold = 0.2, fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", perf_metric_name = "expected_profit", revenue = PROFIT_RESPOND, treatment_cost =COST_TREATMENT)
    #cm_uplift_obj.tradeoff(output=False, n_threads =4)
    modelrateuplift_obj = ModelRateUplift([model.y_true for model in cm_uplift_obj.model_params], cm_uplift_obj.pred_outcome, cm_uplift_obj.e_lift, cm_uplift_obj.feature_mask['isforeign'], \
                                      cm_uplift_obj.spl_params["treatment_cost"],\
                                      cm_uplift_obj.spl_params["revenue"], cm_uplift_obj.proportion_of_interpolation_fitting, 2)

    assert abs(modelrateuplift_obj.harm([0])[0] - 0.014796284418302946) <= 0.001
    assert abs(modelrateuplift_obj.profit([0])[0] - 73633.37972946254) <= 50
    assert abs(modelrateuplift_obj.emp_lift_tr([0])[0] - 0.50814332247557) <= 0.01
    assert abs(modelrateuplift_obj.emp_lift_cn([0])[0] - 0.3188806045090305) <= 0.01
