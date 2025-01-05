# %%
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import seaborn as sns

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from itfs import IterativeFeatureSelector
# %%
seed = 42
np.random.seed(seed)
random.seed(seed)
# %%
############################################
# Create dataset
############################################
def create_df_housing()->pd.DataFrame:
    dataset = fetch_california_housing()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name='target')
    df = pd.concat([y,X], axis=1)
    # Convert some numerical features to categorical features
    cat_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
    for col in cat_cols:
        bin_size = min(30, df[col].nunique())
        df[col] = pd.cut(df[col], bins=bin_size, labels=[f"{col}_{i}" for i in range(bin_size)])

    # Add features with noise
    num_nz_cols = ["AveOccup", "Latitude"]
    for col in num_nz_cols:
        for nz_ratio in [2, 0.01]:
            df[f"{col}_nz_{nz_ratio}"] = df[col] + np.random.normal(0, np.std(df[col].values)*nz_ratio, len(df))
    cat_nz_cols = ["AveOccup"]
    for col in cat_nz_cols:
        for nz_ratio in [0.8, 0.01]:
            nzcol = f"{col}_nz_{nz_ratio}"
            vals = df[col].values.copy()
            nz_size = int(len(df)*nz_ratio)
            nx_idxs = np.random.choice(np.arange(len(df)),nz_size , replace=False)
            nz_vals = np.random.choice(df[col].unique(), nz_size)
            vals[nx_idxs] = nz_vals
            df[nzcol] = vals
    # Add fixed values
    df["fixed1"] = 1
    df["fixed2"] = "a"
    # Add random columns
    df["random1"] = np.random.rand(len(df))
    # Add columns with nearly identical correlation
    df["Population_copy"] = df["Population"].values.copy() 
    # Add columns to ignore but still use
    df["ignore1"] = np.random.rand(len(df))
    # Add columns to force drop
    df["drop1"] = np.random.rand(len(df))
    return df


def create_df_dummy()->pd.DataFrame:
    data_size = 50000
    df = pd.DataFrame({})
    df["x1"] = np.random.rand(data_size)
    df["x2"] = np.random.randn(data_size)
    df["x3"] = np.random.uniform(-0.5,0.5, data_size)
    df["x4"] = np.random.rand(data_size)
    df["x5"] = np.random.randn(data_size)
    df["x6"] = np.random.uniform(-0.5,0.5, data_size)
    for c in df.columns:
        df[c] = (df[c] - np.mean(df[c].values) )/ np.std(df[c].values) 
    df["target"] = np.mean(df.values, axis=1) + np.random.normal(0, 0.01, len(df))

    # Convert some numerical features to categorical features
    cat_cols = ["x1", "x2"]
    for col in cat_cols:
        bin_size = min(30, df[col].nunique())
        df[col] = pd.cut(df[col], bins=bin_size, labels=[f"{col}_{i}" for i in range(bin_size)])

    # Add columns with identical or nearly identical correlation
    df["x4_copy"] = df["x4"].values.copy() 
    df["x5_near"] = df["x5"].values.copy() + np.random.normal(0, 0.001, len(df))

    # Add features with noise
    col = "x6"
    nz_ratio = 100
    df[f"{col}_nz_{nz_ratio}"] = df[col] + np.random.normal(0, np.std(df[col].values)*nz_ratio, len(df))

    col = "x1"
    nz_ratio = 0.8
    nzcol = f"{col}_nz_{nz_ratio}"
    vals = df[col].values.copy()
    nz_size = int(len(df)*nz_ratio)
    nx_idxs = np.random.choice(np.arange(len(df)),nz_size , replace=False)
    nz_vals = np.random.choice(df[col].unique(), nz_size)
    vals[nx_idxs] = nz_vals
    df[nzcol] = vals

    # Add fixed values
    df["fixed1"] = 1
    df["fixed2"] = "a"
    # Add random columns
    df["random1"] = np.random.rand(len(df))
    # Add columns to ignore but still use
    df["ignore1"] = np.random.rand(len(df))
    # Add columns to force drop
    df["drop1"] = np.random.rand(len(df))
    return df
house_df = create_df_housing()
house_df.to_pickle("house_df.pkl")
dummy_df = create_df_dummy()
dummy_df.to_pickle("dummy_df.pkl")