'Retrieve, scale, and set lower and upper bounds for data.'

import pandas as pd
import numpy as np

def load_csv(path:str) -> pd.DataFrame:
    return pd.read_csv(path)

def scale_dataset(df:pd.DataFrame) -> pd.DataFrame:
    """Add scaled x and y data to dataframe."""
    x = df["x"]
    y = df["y"]

    mean_data = df.mean(axis=0)
    std_data = df.std(axis=0)
    df["x_scaled"] = (df["x"] - mean_data["x"]) / std_data["x"]
    df["y_scaled"] = (df["y"] - mean_data["y"]) / std_data["y"]
    return df

def bounds_from_array(X) -> list:
    """Get [(lb, ub), ...] per feature for 1D array.
    
    Args:
        X: input data array 
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    lb = np.min(X, axis=0)
    ub = np.max(X, axis=0)
    return list(zip(lb, ub))