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
# %%
############################################
# Load dataset
############################################
# Choose one and comment out the other

# Dummy data
src_df = create_df_dummy()

# # Housing price data
# src_df = create_df_housing()


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
print_cols("fs.const_cols", fs.const_cols)

# %%
####################
# Remove highly correlated columns
####################
# Calculate correlation
fs.create_cor_df(df=src_df)
sns.heatmap(fs.cor_df, annot=True, fmt=".2f", cmap="coolwarm")

# %%
# Calculate score based on correlation
for cor_th in np.linspace(0.8, 1.0-1e-9, 20):
    fs.fit_corr_cols(df=src_df, cor_th=cor_th, verbose=True)
# Plot correlation trends
fig, error_df = fs.plot_cor_state()

# %%
# Determine columns to drop
cor_error_df = fs.plot_select_cors(
    accept_pth_from_min_error=0.1)
print_cols("fs.drop_cor_cols", fs.drop_cor_cols)

# %%
####################
# Remove columns with high null noise
####################
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
    fs.fit_null_cols(df=src_df, null_p_th=th, verbose=True)
fig, error_df = fs.plot_null_state()
# %%
# Determine columns to drop
null_error_df = fs.plot_select_nulls(accept_pth_from_min_error=0.1)
print_cols("fs.drop_null_cols", fs.drop_null_cols)


# %%
####################
# Remove columns with low feature importance
####################
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
    fs.fit_feature_importance_cols(
        df=src_df, importance_p_th=th, verbose=True)
fig, error_df = fs.plot_feature_importance_state()

# %%
# Determine columns to drop
error_df, fig = fs.plot_select_feature_importance(
    accept_pth_from_min_error=0.1)
print_cols("fs.drop_feature_importance_cols", fs.drop_feature_importance_cols)

# # %%
# import random
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import lightgbm as lgb
# import seaborn as sns

# from sklearn.metrics import root_mean_squared_error
# from sklearn.model_selection import KFold
# from sklearn.datasets import fetch_california_housing
# from sklearn.model_selection import train_test_split

# from itfs import IterativeFeatureSelector
# # %%
# seed = 42
# np.random.seed(seed)
# random.seed(seed)
# # %%
# ############################################
# # Create dataset
# ############################################
# def create_df_housing()->pd.DataFrame:
#     dataset = fetch_california_housing()
#     X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
#     y = pd.Series(dataset.target, name='target')
#     df = pd.concat([y,X], axis=1)
#     # Convert some numerical features to categorical features
#     cat_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
#     for col in cat_cols:
#         bin_size = min(30, df[col].nunique())
#         df[col] = pd.cut(df[col], bins=bin_size, labels=[f"{col}_{i}" for i in range(bin_size)])

#     # Add features with noise
#     num_nz_cols = ["AveOccup", "Latitude"]
#     for col in num_nz_cols:
#         for nz_ratio in [2, 0.01]:
#             df[f"{col}_nz_{nz_ratio}"] = df[col] + np.random.normal(0, np.std(df[col].values)*nz_ratio, len(df))
#     cat_nz_cols = ["AveOccup"]
#     for col in cat_nz_cols:
#         for nz_ratio in [0.8, 0.01]:
#             nzcol = f"{col}_nz_{nz_ratio}"
#             vals = df[col].values.copy()
#             nz_size = int(len(df)*nz_ratio)
#             nx_idxs = np.random.choice(np.arange(len(df)),nz_size , replace=False)
#             nz_vals = np.random.choice(df[col].unique(), nz_size)
#             vals[nx_idxs] = nz_vals
#             df[nzcol] = vals
#     # Add fixed values
#     df["fixed1"] = 1
#     df["fixed2"] = "a"
#     # Add random columns
#     df["random1"] = np.random.rand(len(df))
#     # Add columns with nearly identical correlation

#     df["Population_copy"] = df["Population"].values.copy() 
#     # 無視するが利用するものを追加
#     df["ignore1"] = np.random.rand(len(df))
#     # 強制的に捨てる列を追加
#     df["drop1"] = np.random.rand(len(df))
#     return df


# def create_df_dummy()->pd.DataFrame:
#     data_size = 50000
#     df = pd.DataFrame({})
#     df["x1"] = np.random.rand(data_size)
#     df["x2"] = np.random.randn(data_size)
#     df["x3"] = np.random.uniform(-0.5,0.5, data_size)
#     df["x4"] = np.random.rand(data_size)
#     df["x5"] = np.random.randn(data_size)
#     df["x6"] = np.random.uniform(-0.5,0.5, data_size)
#     for c in df.columns:
#         df[c] = (df[c] - np.mean(df[c].values) )/ np.std(df[c].values) 
#     df["target"] = np.mean(df.values, axis=1) + np.random.normal(0, 0.01, len(df))

#     # 一部の数値特徴をカテゴリ特徴に変換
#     cat_cols = ["x1", "x2"]
#     for col in cat_cols:
#         bin_size = min(30, df[col].nunique())
#         df[col] = pd.cut(df[col], bins=bin_size, labels=[f"{col}_{i}" for i in range(bin_size)])

#     # 相関が一緒orほぼ一緒の列を追加
#     df["x4_copy"] = df["x4"].values.copy() 
#     df["x5_near"] = df["x5"].values.copy() + np.random.normal(0, 0.001, len(df))

#     # ノイズを入れた特徴を追加する
#     col = "x6"
#     nz_ratio = 100
#     df[f"{col}_nz_{nz_ratio}"] = df[col] + np.random.normal(0, np.std(df[col].values)*nz_ratio, len(df))

#     col = "x1"
#     nz_ratio = 0.8
#     nzcol = f"{col}_nz_{nz_ratio}"
#     vals = df[col].values.copy()
#     nz_size = int(len(df)*nz_ratio)
#     nx_idxs = np.random.choice(np.arange(len(df)),nz_size , replace=False)
#     nz_vals = np.random.choice(df[col].unique(), nz_size)
#     vals[nx_idxs] = nz_vals
#     df[nzcol] = vals

#     # 固定の値を追加
#     df["fixed1"] = 1
#     df["fixed2"] = "a"
#     # ランダムな列を追加
#     df["random1"] = np.random.rand(len(df))
#     # df["random2"] = np.random.choice(["a","b","c"], len(df))
#     # df["random2"] = df["random2"].astype("category")
#     # 無視するが利用するものを追加
#     df["ignore1"] = np.random.rand(len(df))
#     # 強制的に捨てる列を追加
#     df["drop1"] = np.random.rand(len(df))
#     return df
# # %%
# ############################################
# # データセットの読み込み
# ############################################
# # 下記どちらかを選んでコメントアウト

# # ダミーデータ
# src_df = create_df_dummy()

# # # 住宅価格データ
# # src_df = create_df_housing()


# # %%
# ####################
# # errorと重要度を計算する関数を用意
# ####################
# kf = KFold(n_splits=5, shuffle=True, random_state=0)
# kf_idxs = [(train_idx, test_idx) for train_idx, test_idx in kf.split(src_df)]
# def calc_error_importance(xdf: pd.DataFrame, ys: np.ndarray)-> tuple[np.ndarray[float], pd.DataFrame]:
#     global kf_idxs
#     oof_preds = []
#     oof_trues = []
#     importance_df = []
#     cv_errors = []
#     for fi, (train_valid_idxs, test_idxs) in enumerate(kf_idxs):
#         train_idxs, valid_idxs = train_test_split(train_valid_idxs, test_size=0.2, random_state=0)
#         train_x = xdf.iloc[train_idxs]
#         train_y = ys[train_idxs]
#         valid_x = xdf.iloc[valid_idxs]
#         valid_y = ys[valid_idxs]
#         test_x = xdf.iloc[test_idxs]
#         test_y = ys[test_idxs]
#         # ハイパーパラメータの設定
#         params = {
#             'objective': 'regression',
#             'metric': 'rmse',
#             'boosting_type': 'gbdt',
#             'learning_rate': 0.1,
#             'num_leaves': 35,
#             'random_state': seed,
#             'verbose': -1
#         }

#         # モデルの学習
#         verbose_eval=0
#         model = lgb.train(
#             params,
#             train_set=lgb.Dataset(train_x, train_y),
#             valid_sets=lgb.Dataset(valid_x, valid_y),
#             num_boost_round=1000,
#             callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True), # early_stopping用コールバック関数
#                                 lgb.log_evaluation(verbose_eval)], # コマンドライン出力用コールバック関数
#         )

#         # テストデータでの予測
#         y_pred = model.predict(test_x, num_iteration=model.best_iteration)
#         # 精度の評価 (RMSE)
#         cv_error = root_mean_squared_error(y_true=test_y,y_pred=y_pred)
#         cv_errors.append(cv_error)
#         # oofに追加
#         oof_preds.extend(y_pred.tolist())
#         oof_trues.extend(test_y.tolist())

#         # 特徴量重要度
#         importance = model.feature_importance(importance_type="gain")
#         feature_names = train_x.columns
#         temp_imp_df = pd.DataFrame({f"importance{fi}": importance})
#         temp_imp_df.index = feature_names
#         importance_df.append(temp_imp_df)
#     importance_df = pd.concat(importance_df, axis=1)
#     oof_error = root_mean_squared_error(y_true=oof_trues, y_pred=oof_preds)
#     cv_errors = np.array(cv_errors)
#     dst_dict = {
#         "oof_error": oof_error,
#         "cv_errors": cv_errors,
#         "importance_df": importance_df
#     }
#     return dst_dict
# # %%
# def print_cols(label, drop_cols):
#     print(f"・{label}", len(drop_cols))
#     for ci, col in enumerate(drop_cols):
#         print(f"{ci+1}: {col}")
#     cols = fs.drop_selected_cols(df=src_df, is_all_true=True).columns
#     print("・残ったcolumns:", len(cols))
#     for col in cols:
#         print(col)
# # %%
# ####################
# # 特徴量選択のインスタンスを生成
# ####################
# fs = IterativeFeatureSelector(
#     calc_error_importance=calc_error_importance,
#     force_drop_cols=["drop1"],
#     ignore_cols=["ignore1"],
#     ycol="target",
#     min_col_size=2,
# )

# # %%
# ####################
# # 値が一定の列を削除
# ####################
# fs.fit_const_cols(df=src_df)
# print_cols("fs.const_cols", fs.const_cols)

# # %%
# ####################
# # 相関が高い列を削除
# ####################
# # 相関を計算
# fs.create_cor_df(df=src_df)
# sns.heatmap(fs.cor_df, annot=True, fmt=".2f", cmap="coolwarm")

# # %%
# # 相関によるスコアを計算
# for cor_th in np.linspace(0.8, 1.0-1e-9, 20):
#     fs.fit_corr_cols(df=src_df, cor_th=cor_th, verbose=True)
# # 相関の様子を描画
# fig, error_df = fs.plot_cor_state()

# # %%
# # 落とす列を決める
# cor_error_df = fs.plot_select_cors(
#     accept_pth_from_min_error=0.1)
# print_cols("fs.drop_cor_cols", fs.drop_cor_cols)

# # %%
# ####################
# # nullノイズの高い列を削除
# ####################
# # null importanceを計算
# fs.create_null_df(df=src_df, null_times=3)
# plt.figure(figsize=(5, 2))
# plt.bar(fs.null_importance_p_df["cols"], fs.null_importance_p_df["p_value"])
# plt.xticks(rotation=90)
# plt.pause(0.1)
# plt.close()

# # %%
# # null importanceのスコアを計算
# for th in np.linspace(0.0, 0.8 , 10):
#     fs.fit_null_cols(df=src_df, null_p_th=th, verbose=True)
# fig, error_df = fs.plot_null_state()
# # %%
# # 落とす列を決める
# null_error_df = fs.plot_select_nulls(accept_pth_from_min_error=0.1)
# print_cols("fs.drop_null_cols", fs.drop_null_cols)


# # %%
# ####################
# # feature importanceで重要度が低い列を削除
# ####################
# # feature importanceを計算
# fs.create_feature_importance_df(df=src_df)
# plt.figure(figsize=(5, 2))
# plt.bar(fs.feature_importance_p_df["cols"], fs.feature_importance_p_df["mean_importance_p"])
# plt.xticks(rotation=90)
# plt.pause(0.1)
# plt.close()
# # %%
# # feature importanceのスコアを計算
# for th in np.linspace(0.0, 0.5, 10):
#     fs.fit_feature_importance_cols(
#         df=src_df, importance_p_th=th, verbose=True)
# fig, error_df = fs.plot_feature_importance_state()

# # %%
# # 落とす列を決める
# error_df, fig = fs.plot_select_feature_importance(
#     accept_pth_from_min_error=0.1)
# print_cols("fs.drop_feature_importance_cols", fs.drop_feature_importance_cols)
