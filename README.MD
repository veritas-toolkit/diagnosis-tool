

# Veritas Toolkit
[![codecov](https://codecov.io/gh/mas-veritas2/veritastool/branch/main/graph/badge.svg?token=0J3QEBHBDU)](https://codecov.io/gh/mas-veritas2/veritastool)
[![PyPI version](https://badge.fury.io/py/veritastool.svg)](https://badge.fury.io/py/veritastool) 
[![Python 3.9](https://img.shields.io/badge/python-3.9-green)](https://www.python.org/downloads/release/python-395/) 
[![Python 3.8](https://img.shields.io/badge/python-3.8-green)](https://www.python.org/downloads/release/python-385/) 
[![Python 3.7](https://img.shields.io/badge/python-3.7-green)](https://www.python.org/downloads/release/python-375/) 
[![GitHub license](https://img.shields.io/github/license/mas-veritas2/veritastool.svg)](https://github.com/mas-veritas2/veritastool/blob/master/license.txt)
[![Python package](https://github.com/mas-veritas2/veritastool/actions/workflows/python-package.yml/badge.svg)](https://github.com/mas-veritas2/veritastool/actions/workflows/python-package.yml) 




<p align="center"><img src="https://raw.githubusercontent.com/mas-veritas2/veritastool/main/icon/veritas_logo_new.png" ></p>


The purpose of this toolkit is to facilitate the adoption of Veritas Methodology on Fairness Assessment and spur industry development. It will also
benefit customers by improving the fairness of financial services delivered by AIDA systems.

  
## Installation

The easiest way to install veritastool is to download it from [`PyPI`](https://pypi.org/project/veritastool/). It's going to install the library itself and its prerequisites as well. It is suggested to create virtual environment with requirements.txt file first.

```python
pip install veritastool
```

Then, you will be able to import the library and use its functionalities. Before we do that, we can run a test function on our sample datasets to see if our codes are performing as expected.

```python
from veritastool.util.utility import test_function_cs
test_function_cs()
```
Output:

<p align="center"><img src="https://raw.githubusercontent.com/mas-veritas2/veritastool/master/icon/test_evaluate_cs.png" width="800" height="100"></p>

### Initialization ##

You can now import the custom library that you would to use for diagnosis. In this example we will use the Credit Scoring custom library. 

```python
from veritastool.model import ModelContainer
from veritastool.fairness import CreditScoring
```

Once the relevant use case object (CreditScoring) and model container (ModelContainer) has been imported, you can upload your contents into the container and initialize the object for diagnosis.

```python

import pickle
import numpy as np

#Load Credit Scoring Test Data
# NOTE: Assume current working directory is the root folder of the cloned veritastool repository
file = "./veritastool/resources/data/credit_score_dict.pickle"
input_file = open(file, "rb")
cs = pickle.load(input_file)

#Reduce into two classes
cs["X_train"]['MARRIAGE'] = cs["X_train"]['MARRIAGE'].replace([0, 3],1)
cs["X_test"]['MARRIAGE'] = cs["X_test"]['MARRIAGE'].replace([0, 3],1)

#Model Container Parameters
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

container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, 
x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,
model_name =  model_name, y_pred= y_pred, y_prob= y_prob)

cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, 
fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", 
perf_metric_name = "balanced_acc", fair_metric_name = "equal_opportunity")                                                      
```
###  API functions ###

There are four main API functions that the user can execute to obtain the fairness diagnosis of their use cases.

**Evaluate**

The evaluate API function computes all performance and fairness metrics and renders it in a table format (default). It
also highlights the primary performance and fairness metrics (automatic if not specified by user).

```python
cre_sco_obj.evaluate()
```
Output:

<p align="center"><img src="https://raw.githubusercontent.com/mas-veritas2/veritastool/master/icon/evaluate-default.png" width="800" height="800"></p>

You can also toggle the widget to view your results in a interactive visualization format.

```python
cre_sco_obj.evaluate(visualize = True)
```
Output:

<p align="center"><img src="https://raw.githubusercontent.com/mas-veritas2/veritastool/master/icon/evaluate_widget.JPG" width="900" height="500"></p>

**Tradeoff**

Computes trade-off between performance and fairness.

```python
cre_sco_obj.tradeoff()
```
Output:

<p align="center"><img src="https://raw.githubusercontent.com/mas-veritas2/veritastool/master/icon/tradeoff.png" width="700" height="600"></p>

** Note: Replace {Balanced Accuracy} with the respective given metrics. 

**Feature Importance**

Computes feature importance of protected features using leave one out analysis.

```python
cre_sco_obj.feature_importance()
```
Output:

<p align="center"><img src="https://raw.githubusercontent.com/mas-veritas2/veritastool/master/icon/feature-imp.png" width="1000" height="500"></p>

** Note: Replace {Balanced Accuracy} & {Equal Opportunity} with the respective given metrics. 

**Compile**

Generates model artifact file in JSON format. This function also runs evaluate(), tradeoff() and feature_importance() if it hasn't already been ran.

```python
cre_sco_obj.compile()
```
Output:

<p align="center"><img src="https://raw.githubusercontent.com/mas-veritas2/veritastool/master/icon/compile.png" width="600" height="200"></p>

**Model Artifact**

A JSON file that stores all the results from evaluate(), tradeoff() and feature_importance().

Output:

<p align="center"><img src="https://raw.githubusercontent.com/mas-veritas2/veritastool/master/icon/json-output.png" width="700" height="800"></p>

## Examples

You may refer to our example notebooks below to see how the toolkit can be applied:

| Filename               | Description      | 
| -----------------------| -------------    | 
| [`CS_Demo.ipynb`](https://github.com/mas-veritas2/veritastool/blob/master/veritastool/examples/CS_demo.ipynb)| Tutorial notebook to diagnose a credit scoring model for predicting customers' loan repayment.            | 
| [`CM_Demo.ipynb`](https://github.com/mas-veritas2/veritastool/blob/master/veritastool/examples/customer_marketing_example/CM_demo.ipynb)          | Tutorial notebook to diagnose a customer marketing uplift model for selecting existing customers for a marketing call to increase the sales of loan product.            | 
| [`nonPythonModel_customMetric_demo.ipynb`](https://github.com/mas-veritas2/veritastool/blob/master/veritastool/examples/nonPythonModel_customMetric_demo.ipynb)          | Tutorial notebook to diagnose a credit scoring model by LibSVM (non-Python) with custom metric.            | 

## License

Veritas Toolkit is licensed under the Apache License, Version 2.0 - see [`LICENSE`](https://raw.githubusercontent.com/mas-veritas2/veritastool/master/license.txt) for more details. 


