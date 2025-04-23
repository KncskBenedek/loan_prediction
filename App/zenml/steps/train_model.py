from zenml import step
from xgboost import XGBClassifier
from collections import Counter
import pandas as pd
@step
def train_model(X_train:pd.DataFrame, y_train:pd.Series):
    count = Counter(y_train)
    scale_pos_weight = count[0] / count[1] 
    return XGBClassifier(scale_pos_weight=scale_pos_weight)