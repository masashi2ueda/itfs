# %%
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
import time
from functools import partial
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
    ''' input text box for float 
    Args:
        label (str): label
        default (float): default value
    Returns:
        float: input value
    '''
    try:
        dst = float(st.text_input(label, default))
    except:
        dst = np.nan
    return dst

def float_text_range_input(
        base_label: str,
        default_start_th: float,
        default_end_th: float,
        default_each_calc_size: int
        )->np.ndarray:
    ''' input text box for float range
    Args:
        base_label (str): label for each input
        default_start_th (float): default start value
        default_end_th (float): default end value
        default_each_calc_size (int): default calc size
    Returns:
        np.ndarray: input value
    '''
    start_th = float_text_input(f"{base_label} start value", default_start_th)
    end_th = float_text_input(f"{base_label} end value", default_end_th)
    each_calc_size = int(float_text_input(f"{base_label} num", default_each_calc_size))
    return np.linspace(start_th, end_th, each_calc_size)

def write_remainer_cols(
        fs: IterativeFeatureSelector,
        src_df: pd.DataFrame,
        drop_cols: list[str]|None = None):
        """ display remaining columns
        Args:
            fs (IterativeFeatureSelector): instance
            src_df (pd.DataFrame): source dataframe
        """
        if drop_cols is not None:
            st.write(f"dropped columns({len(drop_cols)}): {', '.join(drop_cols)}")

        remaind_cols = fs.drop_selected_cols(df=src_df, is_all_true=True).columns
        st.write(f"remaining columns({len(remaind_cols)}): {', '.join(remaind_cols)}")

# %%
def start_server(
        src_df: pd.DataFrame,
        fs_path: str,
        ycol: str,
        calc_error_importance: Callable[[pd.DataFrame, np.ndarray], dict[str, any]],
        force_drop_cols: list[str] = [],
        ignore_cols: list[str] = [],
        min_col_size: int = 2):
    """ start streamlit server
    Args:
        src_df (pd.DataFrame): source dataframe
        fs_path (str): path to save instance
        ycol (str): target column
        calc_error_importance (Callable[[pd.DataFrame, np.ndarray], dict[str, any]]): Function to calculate score and importance.
            see also IterativeFeatureSelector's init docstring.
        force_drop_cols (list[str], optional): columns to drop. Defaults to [].
        ignore_cols (list[str], optional): columns to ignore. Defaults to [].
        min_col_size (int, optional): minimum column size. Defaults to 2.
    """
    # create session state instance
    ps = PageSessionState(__file__)
    ps.set_if_not_exist({"init": False})

    # local functions
    def save_ifs():
        pkl.dump(fs, open(fs_path, "wb"))
    def load_ifs():
        return pkl.load(open(fs_path, "rb"))
    
    def progress_bar(vals: np.ndarray[float], func: Callable[[float], None]):
        # プログレスバーを100%になるまで更新
        # プログレスバーの初期化
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_per_one = []
        for vi, val in enumerate(vals):
            time_start = time.time()
            func(val)
            percent_complete = vi/len(vals)
            progress_bar.progress(percent_complete)
            ptime = time.time() - time_start
            if ptime > 0.1:
                time_per_one.append(ptime)
            status_text.text(f"Progress: {percent_complete*100:.2f}%, Elapsed time: {np.mean(time_per_one):.2f} sec")
        progress_bar.empty()

    # initialize
    if not ps.init:
        if not os.path.exists(fs_path):
            # create instance
            fs = IterativeFeatureSelector(
                force_drop_cols=force_drop_cols,
                calc_error_importance=calc_error_importance,
                ignore_cols=ignore_cols,
                ycol=ycol,
                min_col_size=min_col_size)
            # delete constant columns
            st.write("calculating error without constant columns...")
            fs.fit_const_cols(df=src_df)
            st.write("done.")
            
            save_ifs()
        ps.init = True
    fs = load_ifs()

    ############################
    # constant columns
    ############################
    st.write("---")
    st.header("constant")
    if fs.const_cols is not None:
        write_remainer_cols(fs, src_df, fs.const_cols)
        save_ifs()

    ############################
    # correration columns
    ############################
    st.write("---")
    st.header("correration")
    def show_heat_map():
        fig = plt.figure(figsize=(10, 5))
        sns.heatmap(fs.cor_df, annot=True, fmt=".2f", cmap="coolwarm")
        st.pyplot(fig)
    if fs.cor_df is None:
        if st.button("create correration matrix"):
            st.write("calculatingcorreration matrix...")
            fs.create_cor_df(df=src_df)
            save_ifs()
            st.write("done.")
            show_heat_map()
    else:
        show_heat_map()

    # calculate error with corr each thresould
    th_ranges = float_text_range_input(
        "corr thresould range", 0.5, 1-1e-8, 10)
    if st.button("calculate error with corr each thresould"):
        def fit_corr_cols(th: float):
            fs.fit_corr_cols(df=src_df, cor_th=th, verbose=False)
            save_ifs()
        progress_bar(th_ranges, fit_corr_cols)
    if fs.cor_error_list is not None:
        # Plot correlation trends
        fig, error_df = fs.plot_cor_state()
        st.pyplot(fig)

    # Determine columns to drop
    def drop_corr_cols(th: float):
        cor_error_df, fig = fs.plot_select_cors(
            accept_pth_from_min_error=th)
        st.pyplot(fig)
        write_remainer_cols(fs, src_df, fs.drop_cor_cols)
        save_ifs()
    cor_accept_pth_from_min_error = float_text_input("cor_accept_pth_from_min_error", fs.cor_accept_pth_from_min_error)
    if st.button("drop correration columns"):
        drop_corr_cols(cor_accept_pth_from_min_error)
    elif fs.cor_accept_pth_from_min_error is not None:
        drop_corr_cols(fs.cor_accept_pth_from_min_error)

    ############################
    # null importance columns
    ############################
    st.write("---")
    st.header("null importance")
    null_calc_times = float_text_input("null_calc_times", 3)
    if st.button("create null importance"):
        fs.create_null_df(df=src_df, null_times=int(null_calc_times))
        save_ifs()
    if fs.null_importance_p_df is not None:
        fig = plt.figure(figsize=(5, 2))
        plt.bar(fs.null_importance_p_df["cols"], fs.null_importance_p_df["p_value"])
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # calculate error with corr each thresould
    th_ranges = float_text_range_input(
        "null thresould range", 0.0, 0.5, 10)
    if st.button("calculate error with null importance each thresould"):
        def fit_null_cols(th: float):
            fs.fit_null_cols(df=src_df, null_p_th=th, verbose=False)
            save_ifs()
        progress_bar(th_ranges, fit_null_cols)
    if fs.null_error_list is not None:
        # Plot null trends
        fig, error_df = fs.plot_null_state()
        st.pyplot(fig)

    # Determine columns to drop
    def drop_null_cols(th: float):
        null_error_df, fig = fs.plot_select_nulls(
            accept_pth_from_min_error=th)
        st.pyplot(fig)
        write_remainer_cols(fs, src_df, fs.drop_null_cols)
        save_ifs()
    null_accept_pth_from_min_error = float_text_input("null_accept_pth_from_min_error", fs.null_accept_pth_from_min_error)
    if st.button("drop null columns"):
        drop_null_cols(null_accept_pth_from_min_error)
    elif fs.null_accept_pth_from_min_error is not None:
        drop_null_cols(fs.null_accept_pth_from_min_error)

    ############################
    # feature importance columns
    ############################
    st.write("---")
    st.header("feature importance")
    if st.button("create feature importance"):
        # Calculate feature importance
        fs.create_feature_importance_df(df=src_df)
        save_ifs()
    if fs.feature_importance_p_df is not None:
        fig = plt.figure(figsize=(5, 2))
        plt.bar(fs.feature_importance_p_df["cols"], fs.feature_importance_p_df["mean_importance_p"])
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # calculate error with corr each thresould
    th_ranges = float_text_range_input(
        "importance thresould range", 0.0, 0.5, 10)
    if st.button("calculate error with feature importance each thresould"):
        def fit_importance_cols(th: float):
            fs.fit_feature_importance_cols(df=src_df, importance_p_th=th, verbose=False)
            save_ifs()
        progress_bar(th_ranges, fit_importance_cols)
    if fs.fearute_importance_error_list is not None:
        # Plot null trends
        fig, error_df = fs.plot_feature_importance_state()
        st.pyplot(fig)

    # Determine columns to drop
    def drop_fs_cols(th: float):
        null_error_df, fig = fs.plot_select_feature_importance(
            accept_pth_from_min_error=th)
        st.pyplot(fig)
        write_remainer_cols(fs, src_df, fs.drop_feature_importance_cols)
        save_ifs()
    fearute_importance_accept_pth_from_min_error = float_text_input("fearute_importance_accept_pth_from_min_error", fs.fearute_importance_accept_pth_from_min_error)
    if st.button("drop importance columns"):
        drop_fs_cols(fearute_importance_accept_pth_from_min_error)
    elif fs.fearute_importance_accept_pth_from_min_error is not None:
        drop_fs_cols(fs.fearute_importance_accept_pth_from_min_error)



 
    # st.header("nullノイズの高い列を削除")
    # ranges = float_text_range_input("null", 0.0, 0.3, 10)
    # if st.button("nullノイズの計算"):
    #     for th in ranges:
    #         fs.fit_null_cols(df=src_df, null_p_th=th, error_importance_func=calc_error_importance, verbose=True)
    #     save_ifs()
    # null_accep_pth_from_min_error = float_text_input("null最小エラーからの許容値", fs.null_accept_pth_from_min_error)
    # if st.button("nullノイズの削除"):
    #     # 落とす列を決める
    #     null_error_df, figs = fs.plot_select_nulls(accept_pth_from_min_error=null_accep_pth_from_min_error)
    #     st.write("・fs.drop_null_cols:")
    #     st.write(", ".join(fs.drop_null_cols))
    #     write_remainer_cols(fs, src_df)
    #     for fig in figs:
    #         st.pyplot(fig)
    #     save_ifs()

    # st.header("feature importanceで重要度が低い列を削除")
    # ranges = float_text_range_input("feature importance", 0.0, 1.0, 10)
    # if st.button("feature importanceの計算"):
    #     for th in ranges:
    #         fs.fit_feature_importance_cols(df=src_df, importance_p_th=th, error_importance_func=calc_error_importance, verbose=True)
    #     save_ifs()
    # feature_importance_accep_pth_from_min_error = float_text_input("feature importance最小エラーからの許容値", fs.fearute_importance_accept_pth_from_min_error)
    # if st.button("feature importanceの削除"):
    #     # 落とす列を決める
    #     feature_importance_error_df, figs = fs.plot_select_feature_importance(accept_pth_from_min_error=feature_importance_accep_pth_from_min_error)
    #     st.write("・fs.drop_feature_importance_cols:")
    #     st.write(", ".join(fs.drop_feature_importance_cols))
    #     write_remainer_cols(fs, src_df)
    #     for fig in figs:
    #         st.pyplot(fig)
    #     save_ifs()
    return fs