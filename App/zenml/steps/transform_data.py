import pandas as pd
from sklearn.model_selection import train_test_split
from zenml import step

@step
def transform_data(data: pd.DataFrame, test_size = 0.25, random_state=42 ): 
    data = data.drop("Id")
    X = data.drop("Risk_Flag")
    y = data["Risk_Flag"]
    X_Train, X_Test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state=random_state)
    return X_Train, X_Test, y_train, y_test