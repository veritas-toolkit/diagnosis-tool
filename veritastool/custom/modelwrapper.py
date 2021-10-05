class ModelWrapper(object):
    """
    Abstract Base class to provide an interface that supports non-pythonic models.
    Serves as a template for users to define the
    """

    def __init__(self, model_obj = None, model_file = None, output_file = None):
        """
        Instance attributes
        ----------
        model_obj : object, default=None
                Model object

        model_file : string, default=None
                Path to the model file. e.g. "/home/model.pkl"

        output_file : string, default=None
                Path to which the prediction results will be written to in the form of a csv file. e.g. "/home/results.csv"
        """

        self.model_obj = model_obj
        self.model_file = model_file
        self.output_file = output_file


    def fit(self, x_train, y_train):
        """
        This function is a template for user to specify a custom fit() method that trains the model and saves it to self.model_file.
        An example is as follows:
    
        train_cmd = "train_func --train {x_train} {y_train} {self.model_file}"
        import subprocess
        process = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        Parameters
        -----------
        x_train: pandas Dataframe of shape (m_samples, n_features) or string
                Training dataset. m_samples refers to number of rows in the training dataset.
                The string refers to the dataset path acceptable by the model (e.g. HDFS URI).


        y_train : array of shape (m_samples,)
                Ground truth for training data.
        """
        pass

    def predict(self, x_test):
        """
        This function is a template for user to specify a custom predict() method
        that uses the model saved in self.model_file to make predictions on the test dataset.
    
        Predictions can be either probabilities or labels.
    
        An example is as follows:
    
        pred_cmd = "pred_func --predict {self.model_file} {x_test} {self.output_file}"
        import subprocess
        process = subprocess.Popen(pred_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

        Parameters
        -----------
        x_test : pandas Dataframe or array of shape (n_samples, n_features) or string
                Testing dataset.
                The string refers to the dataset path acceptable by the model (e.g. HDFS URI).

        """
        pass
