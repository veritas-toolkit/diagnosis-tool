class NewMetric: # discuss the case when more than 1 new metric is defined, new class to be created or design to be in dict form
    """
    Base class to add new metrics to the Veritas library.

    Class Attributes
    ------------------
    metric_type: str
        "fair" or "perf"

    metric_group: str
        "classification", "regression" or "uplift"

    metric_name: str
        Name of metric

    metric_definition: str
        Full name of metric

    metric_parity_ratio: str
        "parity" or "ratio"

    enable_flag: boolean
        Whether the new metric can be a primary metric
    """
    metric_type = "fair"
    metric_group = "classification"
    metric_name = "custom base"
    metric_definition = "custom_new_metric base"
    metric_parity_ratio = "parity"
    enable_flag = True

    def compute(self):
        """
        Returns tuple of metric value and privileged group value

        Returns
        -----------
        compute : tuple of floats
            Returns tuple of metric value and privileged group value (applicable if the custom metric is fairness metric)

        """
        #compute function name should be defined add notation for use_case_obj and parameters
        metric_value = 0
        pr_p = 0   #check names later
        return (metric_value, pr_p)
