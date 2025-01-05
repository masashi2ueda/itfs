# %%
import random
import numpy as np
import pandas as pd
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from itfs import IterativeFeatureSelector, DellType
seed = 42
np.random.seed(seed)
random.seed(seed)

# %%
############################################
# Load dataset
############################################
# Choose one and comment out the other

# Dummy data
src_df = pd.read_pickle(open("dummy_df.pkl", "rb"))

# Housing price data
# src_df = pd.read_pickle(open("house_df.pkl", "rb"))


# %%
####################
# Prepare functions to calculate error and importance
####################
kf = KFold(n_splits=5, shuffle=True, random_state=0)
kf_idxs = [(train_idx, test_idx) for train_idx, test_idx in kf.split(src_df)]
def calc_error_importance(xdf: pd.DataFrame, ys: np.ndarray)-> tuple[np.ndarray[float], pd.DataFrame]:
    global kf_idxs
    oof_preds = []
    oof_trues = []
    importance_df = []
    cv_errors = []
    for fi, (train_valid_idxs, test_idxs) in enumerate(kf_idxs):
        train_idxs, valid_idxs = train_test_split(train_valid_idxs, test_size=0.2, random_state=0)
        train_x = xdf.iloc[train_idxs]
        train_y = ys[train_idxs]
        valid_x = xdf.iloc[valid_idxs]
        valid_y = ys[valid_idxs]
        test_x = xdf.iloc[test_idxs]
        test_y = ys[test_idxs]
        # Set hyperparameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 35,
            'random_state': seed,
            'verbose': -1
        }

        # Train the model
        verbose_eval=0
        model = lgb.train(
            params,
            train_set=lgb.Dataset(train_x, train_y),
            valid_sets=lgb.Dataset(valid_x, valid_y),
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True), # Callback for early stopping
                                lgb.log_evaluation(verbose_eval)], # Callback for command-line output
        )

        # Predict on test data
        y_pred = model.predict(test_x, num_iteration=model.best_iteration)
        # Evaluate accuracy (RMSE)
        cv_error = root_mean_squared_error(y_true=test_y,y_pred=y_pred)
        cv_errors.append(cv_error)
        # Add to out-of-fold predictions
        oof_preds.extend(y_pred.tolist())
        oof_trues.extend(test_y.tolist())

        # Feature importance
        importance = model.feature_importance(importance_type="gain")
        feature_names = train_x.columns
        temp_imp_df = pd.DataFrame({f"importance{fi}": importance})
        temp_imp_df.index = feature_names
        importance_df.append(temp_imp_df)
    importance_df = pd.concat(importance_df, axis=1)
    oof_error = root_mean_squared_error(y_true=oof_trues, y_pred=oof_preds)
    cv_errors = np.array(cv_errors)
    dst_dict = {
        "oof_error": oof_error,
        "cv_errors": cv_errors,
        "importance_df": importance_df
    }
    return dst_dict
# %%
def print_cols(label, drop_cols):
    print(f"・{label}", len(drop_cols))
    for ci, col in enumerate(drop_cols):
        print(f"{ci+1}: {col}")
    cols = fs.drop_selected_cols(df=src_df, is_all_true=True).columns
    print("・Remaining columns:", len(cols))
    for col in cols:
        print(col)
# %%
####################
# Generate an instance of feature selection
####################
fs = IterativeFeatureSelector(
    calc_error_importance=calc_error_importance,
    force_drop_cols=["drop1"],
    ignore_cols=["ignore1"],
    ycol="target",
    min_col_size=2,
)

# %%
####################
# Remove columns with constant values
####################
fs.fit_const_cols(df=src_df)
print_cols("const_cols", fs.selected_dropcols[DellType.CONST])

# %%
####################
# Remove highly correlated columns
####################
now_dell_type = DellType.CORR
# Calculate correlation
fs.create_cor_df(df=src_df)
sns.heatmap(fs.cor_df, annot=True, fmt=".2f", cmap="coolwarm")

# %%
# Calculate score based on correlation
for cor_th in np.linspace(0.8, 1.0-1e-9, 20):
    fs.calc_error_each_threshold(df=src_df, delltype=now_dell_type, th=cor_th, verbose=True)
# Plot correlation trends
fig, error_df = fs.plot_state(delltype=now_dell_type)

# %%
# Determine columns to drop
cor_error_df = fs.plot_select(delltype=now_dell_type,accept_th=0.1)
print_cols("fs.drop_cor_cols", fs.selected_dropcols[DellType.CORR])

# %%
####################
# Remove columns with high null noise
####################
now_dell_type = DellType.NULL
# Calculate null importance
fs.create_null_df(df=src_df, null_times=3)
plt.figure(figsize=(5, 2))
plt.bar(fs.null_importance_p_df["cols"], fs.null_importance_p_df["p_value"])
plt.xticks(rotation=90)
plt.pause(0.1)
plt.close()

# %%
# Calculate score based on null importance
for th in np.linspace(0.0, 0.8 , 10):
    fs.calc_error_each_threshold(df=src_df, delltype=now_dell_type, th=th, verbose=True)
fig, error_df = fs.plot_state(delltype=now_dell_type)
# %%
# Determine columns to drop
null_error_df = fs.plot_select(delltype=now_dell_type,accept_th=0.1)
print_cols("fs.drop_null_cols", fs.selected_dropcols[DellType.NULL])


# %%
####################
# Remove columns with low feature importance
####################
now_dell_type = DellType.FEATURE_IMPORTANCE
# Calculate feature importance
fs.create_feature_importance_df(df=src_df)
plt.figure(figsize=(5, 2))
plt.bar(fs.feature_importance_p_df["cols"], fs.feature_importance_p_df["mean_importance_p"])
plt.xticks(rotation=90)
plt.pause(0.1)
plt.close()
# %%
# Calculate score based on feature importance
for th in np.linspace(0.0, 0.5, 10):
    fs.calc_error_each_threshold(
        df=src_df,delltype=now_dell_type, th=th, verbose=True)
fig, error_df = fs.plot_state(delltype=now_dell_type)

# %%
# Determine columns to drop
error_df, fig = fs.plot_select(
    delltype=now_dell_type,accept_th=0.1)
print_cols("fs.drop_feature_importance_cols", fs.selected_dropcols[DellType.FEATURE_IMPORTANCE])

# %%
