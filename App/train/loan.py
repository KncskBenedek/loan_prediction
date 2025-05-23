import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import re
from log_experiment import log_experiment

def loan_predict_experiment():
    data = pd.read_csv("data/Loan Prediction.csv")
    data = data.drop("Id", axis=1)
    X = data.drop("Risk_Flag", axis=1)
    
    y = data["Risk_Flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    pattern = re.compile(r"[^a-zA-Z]")
    for col in cat_cols:
        data[col] = data[col].astype(str).apply(lambda x: pattern.sub("", x))
    
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("encoder", OrdinalEncoder())
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier())
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    tags = {
        "model_type":"XGBoostClassifier",
        "cat_pipeline":"OrdinalEncoder"
    }
    exp_params = {
        "tags":tags,
        "params":None,
        "y_pred":y_pred,
        "y_test":y_test, 
        "experiment_name":"Loan Experiment", 
        "output_type":"binary",
        "model":pipeline,
        "y_pred":y_pred, 
        "artifact_path":"loan_model",
        "X_train":X_train, 
        "model_name":"loan_model"
    }
    log_experiment(**exp_params)

    
loan_predict_experiment()