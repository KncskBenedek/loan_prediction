
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def quic_start():
    
    X, y = datasets.load_iris(return_X_y=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_pred, y_test)
    #log
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment("Iris Quickstart")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        
        sign = infer_signature(X_train, y_pred)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="iris_model",
            signature=sign,
            input_example=X_train,
            registered_model_name="tracking-quickstart"
        )
        print("model_uri:",model_info.model_uri)

quic_start()