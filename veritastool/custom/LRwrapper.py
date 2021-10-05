from .modelwrapper import ModelWrapper
import numpy as np
from imblearn.over_sampling import SMOTENC
from sklearn.preprocessing import StandardScaler
import time

class LRwrapper(ModelWrapper):
    """
    Abstract Base class to provide an interface that supports non-pythonic models.
    Serves as a template for users to define the

    """

    def __init__(self, model_obj):
        self.model_obj = model_obj
        #self.output_file = output_file
       
    """
    Parameters
    ----------
    model_file : string
            Path to the model file. e.g. "/home/model.pkl"

    output_file : string
            Path to which the prediction results will be written to in the form of a csv file. e.g. "/home/results.csv"
    """

    def fit(self, X, y):
    
        #verbose and print('upsampling...')
        categorical_features = [i for i, col in enumerate(X.columns) if X[col].dtype == 'int8']
        smote = SMOTENC(random_state=0, categorical_features=categorical_features)
        X, y = smote.fit_resample(X, y)

        #verbose and print('scaling...')
        scaling = StandardScaler()
        X = scaling.fit_transform(X)

        #verbose and print('fitting...')
        #verbose and print('C:', C)
        #model = LogisticRegression(random_state=seed, C=C, max_iter=4000)
        
        
        self.model_obj.fit(X, y)
        #time.sleep(30)
        
        """
        This function is a template for user to specify a custom fit() method that trains the model and saves it to self.model_file.
        An example is as follows:
    
        train_cmd = "train_func --train {x_train} {y_train} {self.model_file}"
        import subprocess
        process = subprocess.Popen(train_cmd.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()

    """
        #pass

    def predict(self, x_test, best_th = 0.43):
        test_probs = self.model_obj.predict_proba(x_test)[:, 1] 
        test_preds = np.where(test_probs > best_th, 1, 0)
        return test_preds

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
        #pass
