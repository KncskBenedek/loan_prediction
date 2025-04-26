import mlflow
from mlflow.models import infer_signature
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

def log_experiment(
                   model_type:str,
                   y_pred, 
                   y_test, 
                   params=None, 
                   uri:str="http://127.0.0.1:5000", 
                   experiment_name:str="Quickstart", 
                   output_type:str="binary" 
                   ):
    
    
    mlflow.set_tracking_uri(uri=uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        if params is not None: mlflow.log_params(params)
        mlflow.set_tag("model_type", model_type)
        mlflow.log_metric("accuracy", accuracy_score(y_pred, y_test))
        mlflow.log_metric("precision", precision_score(y_test, y_pred, average=output_type))
        mlflow.log_metric("recall", recall_score(y_test, y_pred, average=output_type))
        mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average=output_type))

def log_model(model,y_pred, artifact_path, X_train, model_name):

    sign = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(
            sk_model=model,
            signature=sign,
            artifact_path=artifact_path,
            input_example=X_train,
            registered_model_name=model_name
        )
