import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder


def preprocess_data(path):
    df = pd.read_csv(path)

    df.repalce("?", np.nan,inplace=True)
    df.dropna(inplace=True)

    #Convert numeric
    for col in df.columns:
        try:
            df[col] = df[col].astype(float)
        except:
            pass
    # Encode categorical
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    return df


