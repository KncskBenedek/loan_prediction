
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from log_experiment import log_experiment

def quick_start():
    
    X, y = datasets.load_iris(return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    params = {"criterion":"entropy", "max_depth":3}
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    log_experiment(model=model,
                   model_name="iris_model",
                   model_type="DecisionTreeClassifier", 
                   artifact_path="iris_model",
                   params=params,
                   X_train=X_train, 
                   X_test=X_test,
                   y_test=y_test, 
                   experiment_name="Iris Quickstart", 
                   output_type="weighted")
        

quick_start()