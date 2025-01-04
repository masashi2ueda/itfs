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

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from itfs import IterativeFeatureSelector
else:
    from .feature_select import IterativeFeatureSelector

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
def start_server(
        src_df: pd.DataFrame,
        ifs_path: str,
        ycol: str,
        calc_error_importance: Callable[[pd.DataFrame, np.ndarray], tuple[list[float], pd.DataFrame]],
        calc_error: Callable[[pd.DataFrame, np.ndarray], list[float]],
        force_drop_cols: list[str] = [],
        ignore_cols: list[str] = [],
        min_col_size: int = 2):
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
    if fs.cor_df is None:
        if st.button("相関行列の計算"):
            fs.create_cor_df(src_df)
    else:
        st.write("・fs.cor_error_df:")
        fig = plt.figure(figsize=(10, 10))
        sns.heatmap(fs.cor_df, annot=True, fmt=".2f", cmap="coolwarm")

        

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
    return fs