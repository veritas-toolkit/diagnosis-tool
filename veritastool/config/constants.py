from os.path import dirname, join

class Constants:
    """
    Define fixed parameters in this class
    """
    def __init__(self):
        """
        Instance attributes
        ----------------------
        fair_threshold_low : float
                Fairness threshold minimum value

        fair_threshold_high : float
                Fairness threshold maximum value

        fair_neutral_threshold_low : float
                Fairness neutral threshold minumum value

        fair_neutral_threshold_high : float
                Fairness neutral threshold maximum value

        proportion_of_interpolation_fitting_low : float
                Proportion of interpolation fitting minumum value

        proportion_of_interpolation_fitting_high : float
                Proportion of interpolation fitting maximum value

        selection_threshold : float
                Selection threshold

        k : int
                Number of samples to calculate confidence interval

        array_size : int
                To be used in performance_dynamics() to determine size of samples

        decimals : int
                Number of decimal places for values to be rounded off
        """
        import configparser
        file = join(dirname(__file__), 'config.ini')
        config = configparser.ConfigParser()
        config.read(file)
        self.fair_threshold_low = config.getfloat('threshold', 'fair_threshold_low')
        self.fair_threshold_high  = config.getfloat('threshold', 'fair_threshold_high')
        self.fair_neutral_threshold_low = config.getfloat('threshold', 'fair_neutral_tolerance_low')
        self.fair_neutral_threshold_high = config.getfloat('threshold', 'fair_neutral_tolerance_high')
        self.proportion_of_interpolation_fitting_low = config.getfloat('threshold', 'proportion_of_interpolation_fitting_low')
        self.proportion_of_interpolation_fitting_high = config.getfloat('threshold', 'proportion_of_interpolation_fitting_high')
        self.selection_threshold = config.getfloat('threshold', 'selection_threshold')
        self.k = config.getint('default', 'k')
        self.array_size = config.getint('default', 'array_size')
        self.decimals = config.getint('default', 'decimals')
