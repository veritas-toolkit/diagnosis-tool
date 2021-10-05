from .newmetric import *
class NewMetricChild(NewMetric): # discuss the case when more than 1 new metric is defined, new class to be created or design to be in dict form
    """
    Base class to add new metrics to the Veritas library.

    Class Attributes
    ------------------
    metric_type: string
        "fair" or "perf"

    metric_group: string
        "classification", "regression" or "uplift"

    metric_name: string
        Name of metric

    metric_definition: string
        Full name of metric

    metric_parity_ratio: string
        "parity" or "ratio"

    enable_flag: boolean
    """

    metric_type = "fair"
    metric_group = "classification"
    metric_name = "custom_new_metric"
    metric_definition = "custom_new_metric child"
    metric_parity_ratio = "parity"
    enable_flag = True


    def compute(): #compute function name should be defined add notation for use_case_obj and parameters
        """
         Compute function for new metrics

         Returns
         -----------
         compute : tuple
             Returns tuple of metric value and privileged group value
         """

        metric_value = 0
        pr_p = 0
        
        pass
        return (metric_value, pr_p)    #check names later
    
    
#check enable flag i fariness metric and performance metrics    