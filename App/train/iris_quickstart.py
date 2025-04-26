
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from log_experiment import log_experiment, log_model

def quick_start():
    
    X, y = datasets.load_iris(return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    params = {"criterion":"entropy", "max_depth":3}
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    tags = {
        "model_type":"DecisionTreeClassifier",
        "test":"test"
    }
    log_experiment(
                   tags=tags,
                   params=params,
                   y_pred=y_pred,
                   y_test=y_test, 
                   experiment_name="Iris Quickstart", 
                   output_type="weighted")
    log_model(model=model, y_pred=y_pred, artifact_path="iris_model",X_train=X_train, model_name="iris_model")

quick_start()