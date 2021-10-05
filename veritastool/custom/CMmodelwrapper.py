from .modelwrapper import ModelWrapper
import time

class CMmodelwrapper(ModelWrapper):
    """
    Abstract Base class to provide an interface that supports non-pythonic models.
    Serves as a template for users to define the

    """

    def __init__(self, model_obj = None, model_file = None, output_file = None):
        super().__init__(model_obj, model_file, output_file)

    """
    Parameters
    ----------
    model_file : string
            Path to the model file. e.g. "/home/model.pkl"

    output_file : string
            Path to which the prediction results will be written to in the form of a csv file. e.g. "/home/results.csv"
    """

    def fit(self, X, y):

        """
        This function is a template for user to specify a custom fit() method that trains the model and saves it to self.model_file.
        An example is as follows:
    
        train_cmd = "train_func --train {x_train} {y_train} {self.model_file}"
        import subprocess
        process = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    """
        self.model_obj.fit(X, y)
        time.sleep(30)

    def predict_proba(self, X):

        """
        This function is a template for user to specify a custom predict() method
        that uses the model saved in self.model_file to make predictions on the test dataset.
    
        Predictions can be either probabilities or labels.
    
        An example is as follows:
    
        pred_cmd = "pred_func --predict {self.model_file} {x_test} {self.output_file}"
        import subprocess
        process = subprocess.Popen(pred_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    """
        return self.model_obj.predict_proba(X)

