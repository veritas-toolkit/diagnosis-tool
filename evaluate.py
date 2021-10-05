import numpy as np
import pandas as pd
import datetime
import json
from ..utility import *
from .fairness_metrics import FairnessMetrics
from ..performance_metrics import PerformanceMetrics
from ..custom.newmetric import *
from ..custom.newmetric_child import *
from ..fairness.tradeoff import TradeoffRate
import ipywidgets as widgets
import IPython
from ipywidgets import Layout, Button, Box, VBox, HBox, Text, GridBox
from IPython.display import display, clear_output, HTML
from IPython.core.display import HTML
import matplotlib.pyplot as plt
import os
import time
from memory_profiler import profile

class Fairness:
    """
    Base Class with attributes used across all use cases within Machine Learning model fairness evaluation.
    """
    def __init__(self, model_params):
        """
        Instance Attributes
        ------------------------
        model_params : object of type ModelContainer
                Data holder that contains all the attributes of the model to be assessed. Compulsory input for initialization.

        fair_metric_obj : object
                Stores the FairnessMetrics() object and contains the result of the computations.

        perf_metric_obj : object
                Stores the PerformanceMetrics() object and contains the result of the computations.

        percent_distribution : dictionary, default = None
                Stores the percentage breakdown of the classes in y_true.

        calibration_score : float, default = None
                The brier score loss computed for calibration. Computable if y_prob is given.

        tradeoff_obj : object
                Stores the TradeoffRate() object and contains the result of the computations.

        correlation_output : Dataframe, default = None
                Pairwise correlation of most important features (top 20 feature + protected variables).

        feature_mask : dictionary of lists, default = None
                Stores the mask array for every protected variable applied on the x_test dataset.

        fair_conclusion : dictionary
                Contains conclusion of how the primary fairness metric compares against the fairness threshold. The key will be the protected variable and the conclusion will be "fair" or "unfair".
                e.g. {"gender": {'fairness_conclusion': "fair", "threshold": 0.01}, "race":{'fairness_conclusion': "unfair", "threshold": 0.01}}

        evaluate_status : int, default=0
                Tracks the status of the completion of the evaluate() method to be checked in compile(). Either 1 for complete or -1 for error if any exceptions were raised.

        evaluate_status_cali: boolean, default=False
                Tracks the status of the completion of the calibration curve step within evaluate() method to be checked in compile().
                False = Skipped (if y_prob is not provided)
                True = Complete

        tradeoff_status : int, default=0
                Tracks the status of the completion of the tradeoff() method to be checked in compile().
                0 = Not started
                1 = Complete
                -1 = Skipped (if y_prob is not provided)

        feature_imp_status : int, default=0
                Tracks the status of the completion of the compute_feature_imp() method to be checked in compile(). Either 1 for complete or -1 for error if any exceptions were raised.

        feature_imp_status_loo: boolean, default=False
                Tracks the status of the completion of the leave-one-out analysis step within feature_importance() method to be checked in compile().
                False = Skipped (if x_train or y_train or model object or fit/predict operator names are not provided)
                True = Complete

        feature_imp_status_corr: boolean, default=False
                Tracks the status of the completion of the correlation matrix computation step within feature_importance() method to be checked in compile().
                False = Skipped (if the correlation dataframe is not provided in ModelContainer)
                True = Complete

        feature_imp_values: dictionary of lists, default = None
                Contains the difference in metric values between the original and loco models for each protected variable.

                {"gender":
                         {
                          "gender": (perf_delta, fair_delta, flip, suggestion),
                          "race": (perf_delta, fair_delta, flip, suggestion)
                          },
                "race":
                         {
                          "gender": (perf_delta, fair_delta, flip, suggestion),
                          "race": (perf_delta, fair_delta, flip, suggestion)
                          }
                }

                flip = "fair to fair", "unfair to fair", "fair to unfair", "unfair to unfair"

        """
        self.model_params = model_params
        self.fair_metric_obj = None
        self.perf_metric_obj = None
        self.percent_distribution = None
        self.calibration_score = None
        self.calibration_curve_bin = None
        self.tradeoff_obj = None
        self.correlation_output = None
        self.feature_mask = self._set_feature_mask()
        self.fair_conclusion = None
        self.evaluate_status = 0
        self.tradeoff_status = 0
        self.feature_imp_status = 0
        self.feature_imp_values = None
        self.feature_imp_status_corr = False
        self.feature_imp_status_loo = False
    @profile
    def evaluate(self, visualize=False, output=True):
        """
        Computes the percentage count of subgroups, performance, and fairness metrics together with their confidence intervals, calibration score & fairness metric self.fair_conclusion for all protected variables.

        Parameters
        ----------
        visualize : boolean
                If visualize = True, output will be overwritten to False and run fairness_widget() from Fairness.

        output : boolean
                If output = True, _print_evaluate() from Fairness will run.

        Returns
        ----------
        _fairness_widget() or _print_evaluate()
        """
        self._compute_performance()
        self._compute_fairness()
        self._fairness_conclusion()
        self.evaluate_status = 1

        
        if visualize == True:
            output = False
            self._fairness_widget()
        
        if output == True:
            self._print_evaluate()
        