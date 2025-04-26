import pandas as pd
import numpy as np
import re
data = pd.read_csv("data/Loan Prediction.csv")

cat_cols = data.select_dtypes(include='object').columns.tolist()
pattern = re.compile(r"[^a-zA-Z]")
clear_col_vals = data[col].astype(str).apply(lambda x: pattern.sub("", x))
for col in cat_cols:
    clear_col_vals = data[col].astype(str).apply(lambda x: pattern.sub("", x))
    print(np.unique(clear_col_vals))