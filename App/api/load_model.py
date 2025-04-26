import mlflow
def load_prod_model(model, version="latest"):
    return mlflow.pyfunc.load_model(f"models:/{model}/{version}")
