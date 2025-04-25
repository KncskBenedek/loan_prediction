import mlflow
def load_prod_model():
    return mlflow.sklearn.load_model("models:/tracking-quickstart/latest")
