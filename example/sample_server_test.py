import streamlit as st
import matplotlib.pyplot as plt
from pgss import PageSessionState
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle as pkl
import os

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


import streamlit as st
import matplotlib.pyplot as plt
from pgss import PageSessionState
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
import pickle as pkl
import os
import seaborn as sns

from typing import Callable
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


from itfs import IterativeFeatureSelector
# %%
############################################
# データセットの作成
############################################
def create_df_housing()->pd.DataFrame:
    dataset = fetch_california_housing()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name='target')
    df = pd.concat([y,X], axis=1)
    # 一部の数値特徴をカテゴリ特徴に変換
    cat_cols = ["MedInc", "HouseAge", "AveRooms", "AveBedrms"]
    for col in cat_cols:
        bin_size = min(30, df[col].nunique())
        df[col] = pd.cut(df[col], bins=bin_size, labels=[f"{col}_{i}" for i in range(bin_size)])

    # ノイズを入れた特徴を追加する
    num_nz_cols = ["Population","AveOccup", "Latitude", "Longitude"]
    for col in num_nz_cols:
        for nz_ratio in [2, 0.01]:
            df[f"{col}_nz_{nz_ratio}"] = df[col] + np.random.normal(0, np.std(df[col].values)*nz_ratio, len(df))
    cat_nz_cols = ["Population","AveOccup"]
    for col in cat_nz_cols:
        for nz_ratio in [0.8, 0.01]:
            nzcol = f"{col}_nz_{nz_ratio}"
            vals = df[col].values.copy()
            nz_size = int(len(df)*nz_ratio)
            nx_idxs = np.random.choice(np.arange(len(df)),nz_size , replace=False)
            nz_vals = np.random.choice(df[col].unique(), nz_size)
            vals[nx_idxs] = nz_vals
            df[nzcol] = vals
    # 固定の値を追加
    df["fixed1"] = 1
    df["fixed2"] = "a"
    # ランダムな列を追加
    df["random1"] = np.random.rand(len(df))
    # 相関がほぼ一緒の列を追加
    df["Population_copy"] = df["Population"].copy() 
    # 無視するが利用するものを追加
    df["ignore1"] = np.random.rand(len(df))
    # 強制的に捨てる列を追加
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

    # 一部の数値特徴をカテゴリ特徴に変換
    cat_cols = ["x1", "x2"]
    for col in cat_cols:
        bin_size = min(30, df[col].nunique())
        df[col] = pd.cut(df[col], bins=bin_size, labels=[f"{col}_{i}" for i in range(bin_size)])

    # 相関が一緒orほぼ一緒の列を追加
    df["x4_copy"] = df["x4"].values.copy() 
    df["x5_near"] = df["x5"].values.copy() + np.random.normal(0, 0.001, len(df))

    # ノイズを入れた特徴を追加する
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

    # 固定の値を追加
    df["fixed1"] = 1
    df["fixed2"] = "a"
    # ランダムな列を追加
    df["random1"] = np.random.rand(len(df))
    # df["random2"] = np.random.choice(["a","b","c"], len(df))
    # df["random2"] = df["random2"].astype("category")
    # 無視するが利用するものを追加
    df["ignore1"] = np.random.rand(len(df))
    # 強制的に捨てる列を追加
    df["drop1"] = np.random.rand(len(df))
    return df

@st.cache_data
def get_src_df():
    src_df = create_df_housing()
    # src_df = create_df_dummy()
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    kf_idxs = [(train_idx, test_idx) for train_idx, test_idx in kf.split(src_df)]
    return src_df, kf_idxs

# %%
src_df, kf_idxs = get_src_df()



# %%
####################
# errorと重要度を計算する関数を用意
####################
def calc_error_importance(xdf: pd.DataFrame, ys: np.ndarray)-> tuple[np.ndarray[float], pd.DataFrame]:
    global kf_idxs
    oof_errors = []
    importance_df = []
    for fi, (train_valid_idxs, test_idxs) in enumerate(kf_idxs):
        train_idxs, valid_idxs = train_test_split(train_valid_idxs, test_size=0.2, random_state=0)
        train_x = xdf.iloc[train_idxs]
        train_y = ys[train_idxs]
        valid_x = xdf.iloc[valid_idxs]
        valid_y = ys[valid_idxs]
        test_x = xdf.iloc[test_idxs]
        test_y = ys[test_idxs]
        # ハイパーパラメータの設定
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.1,
            'num_leaves': 80,
            'verbose': -1
        }

        # モデルの学習
        verbose_eval=0
        model = lgb.train(
            params,
            train_set=lgb.Dataset(train_x, train_y),
            valid_sets=lgb.Dataset(valid_x, valid_y),
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True), # early_stopping用コールバック関数
                                lgb.log_evaluation(verbose_eval)], # コマンドライン出力用コールバック関数
        )

        # テストデータでの予測
        y_pred = model.predict(test_x, num_iteration=model.best_iteration)

        # 精度の評価 (RMSE)
        errors = np.sqrt((y_pred - test_y) ** 2)
        oof_errors.extend(errors)

        # 特徴量重要度
        importance = model.feature_importance(importance_type="gain")
        feature_names = train_x.columns
        temp_imp_df = pd.DataFrame({f"importance{fi}": importance})
        temp_imp_df.index = feature_names
        importance_df.append(temp_imp_df)
    importance_df = pd.concat(importance_df, axis=1)
    oof_errors = np.array(oof_errors)
    return oof_errors, importance_df
def calc_error(xdf: pd.DataFrame, ys: np.ndarray) -> np.ndarray[float]:
    return calc_error_importance(xdf=xdf, ys=ys)[0]
# %%
import numpy as np
vals1 = np.random.rand(100)
vals2 = np.random.rand(100)
np.sqrt((vals1 - vals2) ** 2)
# %%
def float_text_input(label:str, default:float)->float:
    try:
        dst = float(st.text_input(label, default))
    except:
        dst = np.nan
    return dst

def float_text_range_input(
        tag: str,
        default_start_th: float,
        default_end_th: float,
        default_each_calc_size: int
        )->np.ndarray:
    start_th = float_text_input(f"{tag}のスタート閾値", default_start_th)
    end_th = float_text_input(f"{tag}のエンド閾値", default_end_th)
    each_calc_size = int(float_text_input(f"{tag}の計算回数", default_each_calc_size))
    return np.linspace(start_th, end_th, each_calc_size)

def write_remainer_cols(fs: IterativeFeatureSelector, src_df: pd.DataFrame):
        remaind_cols = fs.drop_selected_cols(df=src_df, is_all_true=True).columns
        st.write("・残ったcolumns:", len(remaind_cols), "個")
        st.write(", ".join(remaind_cols))

# %%
# def start_server(
#         src_df: pd.DataFrame,
#         ifs_path: str,
#         ycol: str,
#         calc_error_importance: Callable[[pd.DataFrame, np.ndarray], tuple[list[float], pd.DataFrame]],
#         calc_error: Callable[[pd.DataFrame, np.ndarray], list[float]],
#         force_drop_cols: list[str] = [],
#         ignore_cols: list[str] = [],
#         min_col_size: int = 2):
ifs_path="ifs.pkl"
ycol="target"
force_drop_cols: list[str] = []
ignore_cols: list[str] = []
min_col_size = 2
ps = PageSessionState(__file__)
ps.set_if_not_exist({"init": False})
def save_ifs():
    pkl.dump(fs, open(ifs_path, "wb"))
def load_ifs():
    return pkl.load(open(ifs_path, "rb"))
if not ps.init:
    if not os.path.exists(ifs_path):
        # instanceを作成
        fs = IterativeFeatureSelector(
            force_drop_cols=force_drop_cols,
            ignore_cols=ignore_cols,
            ycol=ycol,
            min_col_size=min_col_size)
        # 固定値列の判定は最初にやる
        fs.fit_const_cols(df=src_df)
        save_ifs()
    ps.init = True
fs = load_ifs()

############################
st.header("固定列を削除")
if fs.const_cols is not None:
    st.write("・fs.const_cols:")
    st.write(", ".join(fs.const_cols))
    write_remainer_cols(fs, src_df)
    save_ifs()

############################
st.header("相関が高い列を削除")
# 相関行列を作成
def show_cor_df():
    st.write("・相関行列:")
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(fs.cor_df, annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(fig)
if fs.cor_df is None:
    if st.button("相関行列の作成"):
        fs.create_cor_df(src_df)
        save_ifs()
        show_cor_df()
else:
    show_cor_df()

# 格相関を削除した時のエラーを計算
ranges = float_text_range_input("相関", 0.5, 1.0, 10)
if st.button("相関の計算"):
    for cor_th in ranges:
        fs.fit_corr_cols(df=src_df, cor_th=cor_th, error_func=calc_error, verbose=True)
    save_ifs()

cor_accep_pth_from_min_error = float_text_input("cor最小エラーからの許容値", fs.cor_accept_pth_from_min_error)
if st.button("相関の削除"):
    # 落とす列を決める
    cor_error_df, figs = fs.plot_select_cors(accept_pth_from_min_error=cor_accep_pth_from_min_error)
    st.write("・fs.drop_cor_cols:")
    st.write(", ".join(fs.drop_cor_cols))
    write_remainer_cols(fs, src_df)
    for fig in figs:
        st.pyplot(fig)
    save_ifs()

st.header("nullノイズの高い列を削除")
ranges = float_text_range_input("null", 0.0, 0.3, 10)
if st.button("nullノイズの計算"):
    for th in ranges:
        fs.fit_null_cols(df=src_df, null_p_th=th, error_importance_func=calc_error_importance, verbose=True)
    save_ifs()
null_accep_pth_from_min_error = float_text_input("null最小エラーからの許容値", fs.null_accept_pth_from_min_error)
if st.button("nullノイズの削除"):
    # 落とす列を決める
    null_error_df, figs = fs.plot_select_nulls(accept_pth_from_min_error=null_accep_pth_from_min_error)
    st.write("・fs.drop_null_cols:")
    st.write(", ".join(fs.drop_null_cols))
    write_remainer_cols(fs, src_df)
    for fig in figs:
        st.pyplot(fig)
    save_ifs()

st.header("feature importanceで重要度が低い列を削除")
ranges = float_text_range_input("feature importance", 0.0, 1.0, 10)
if st.button("feature importanceの計算"):
    for th in ranges:
        fs.fit_feature_importance_cols(df=src_df, importance_p_th=th, error_importance_func=calc_error_importance, verbose=True)
    save_ifs()
feature_importance_accep_pth_from_min_error = float_text_input("feature importance最小エラーからの許容値", fs.fearute_importance_accept_pth_from_min_error)
if st.button("feature importanceの削除"):
    # 落とす列を決める
    feature_importance_error_df, figs = fs.plot_select_feature_importance(accept_pth_from_min_error=feature_importance_accep_pth_from_min_error)
    st.write("・fs.drop_feature_importance_cols:")
    st.write(", ".join(fs.drop_feature_importance_cols))
    write_remainer_cols(fs, src_df)
    for fig in figs:
        st.pyplot(fig)
    save_ifs()
# return fs
fs


# # %%

# fs = start_server(src_df=src_df, ifs_path="ifs.pkl",
# ycol="target", calc_error_importance=calc_error_importance,
# calc_error=calc_error)
# fs