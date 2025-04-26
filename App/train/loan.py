import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from log_experiment import log_experiment, log_model

def loan_predict_experiment():
    data = pd.read_csv("data/Loan Prediction.csv")
    data = data.drop("Id", axis=1)
    X = data.drop("Risk_Flag", axis=1)
    y = data["Risk_Flag"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
    """ params={
        "n_estimators": 1000,
        "min_samples_split":10,
        "min_samples_leaf":1,
        "max_features":"log2",
        "max_depth":70,
        "bootstrap":False
    } """
    
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("encoder", OneHotEncoder())
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier())
    ])
    print("fit")
    pipeline.fit(X_train, y_train)
    print("log")
    y_pred = pipeline.predict(X_test)
    log_experiment(
                   model_type="XGBoostClassifier",
                   y_pred=y_pred,
                   y_test=y_test, 
                   params=None, 
                   experiment_name="Loan experiment",
                   output_type="binary")
    log_model(pipeline, y_pred, "loan_model", X_train, "loan_model")
loan_predict_experiment()