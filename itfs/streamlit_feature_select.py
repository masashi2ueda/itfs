# %%
import streamlit as st
import matplotlib.pyplot as plt
from pgss import PageSessionState
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os
import seaborn as sns
import time
from typing import Callable
import yaml

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from itfs import IterativeFeatureSelector
else:
    from .feature_select import IterativeFeatureSelector, DellType

VERBOSE = False
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
        dell_type: DellType,
        drop_cols: list[str]|None = None):
        """ display remaining columns
        Args:
            fs (IterativeFeatureSelector): instance
            src_df (pd.DataFrame): source dataframe
        """
        if drop_cols is not None:
            st.write(f"dropped columns({len(drop_cols)}): {', '.join(drop_cols)}")
        orless_dell_types = fs._get_orless_delltype(delltype=dell_type)
        remaind_cols = fs.drop_selected_cols(df=src_df, is_drop_ycol=orless_dell_types).columns
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
    def save_ifs(fs: IterativeFeatureSelector):
        pkl.dump(fs, open(fs_path, "wb"))
        dst_dict = fs.create_dst_dict()
        with open(fs_path.replace(".pkl", ".yaml"), "w") as f:
            yaml.dump(dst_dict, f)

    def load_ifs():
        return pkl.load(open(fs_path, "rb"))
    
    def progress_bar(vals: np.ndarray[float], func: Callable[[float], None]):
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
            mean_time = 0 if len(time_per_one) == 0 else np.mean(time_per_one)
            status_text.text(f"Progress: {percent_complete*100:.2f}%, Elapsed time: {mean_time:.2f} sec")
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
            save_ifs(fs)
        ps.init = True
    fs = load_ifs()

    def fit_select(
            base_name: str,
            delltype: DellType,
            default_range_start: float,
            default_range_end: float,
            default_each_calc_size: int):
        # calculate error with corr each thresould
        th_ranges = float_text_range_input(
            f"{base_name} thresould range",
            default_range_start,
            default_range_end,
            default_each_calc_size)
        if st.button(f"calculate error with {base_name} each thresould"):
            def core_error_func(th: float):
                fs.calc_error_each_threshold(
                    df=src_df,
                    delltype=delltype, th=th, verbose=VERBOSE)
                save_ifs(fs)
            progress_bar(th_ranges, core_error_func)
        # show state
        delltype_error_list = [item for item in fs.error_list if item["delltype"] == delltype]
        if len(delltype_error_list) != 0:
            # Plot correlation trends
            fig, error_df = fs.plot_state(delltype=delltype)
            st.pyplot(fig)
    
        # Determine columns to drop
        def drop_cols(th: float):
            cor_error_df, fig = fs.plot_select(delltype=delltype, accept_th=th)
            st.pyplot(fig)
            write_remainer_cols(fs=fs, src_df=src_df, dell_type=delltype, drop_cols=fs.selected_dropcols[delltype])
            save_ifs(fs)
        delltype_accept_th =  fs.accept_ths[delltype]
        accept_th = float_text_input(f"{base_name} accept th", delltype_accept_th)
        if st.button(f"drop {base_name} columns"):
            drop_cols(accept_th)
        elif delltype_accept_th is not None:
            drop_cols(delltype_accept_th)

    ############################
    # constant columns
    ############################
    now_dell_type = DellType.CONST
    st.write("---")
    st.header("constant")
    const_cols = fs.selected_dropcols[now_dell_type]
    if const_cols is not None:
        write_remainer_cols(
            fs=fs, src_df=src_df, drop_cols=const_cols, dell_type=now_dell_type)

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
            save_ifs(fs)
            st.write("done.")
            show_heat_map()
    else:
        show_heat_map()
    fit_select(
        base_name="correration",
        delltype=DellType.CORR,
        default_range_start=0.5,
        default_range_end=1-1e-8,
        default_each_calc_size=10)


    ############################
    # null importance columns
    ############################
    st.write("---")
    st.header("null importance")
    null_calc_times = float_text_input("null_calc_times", 3)
    if st.button("create null importance"):
        fs.create_null_df(df=src_df, null_times=int(null_calc_times))
        save_ifs(fs)
    if fs.null_importance_p_df is not None:
        fig = plt.figure(figsize=(5, 2))
        plt.bar(fs.null_importance_p_df["cols"], fs.null_importance_p_df["p_value"])
        plt.xticks(rotation=90)
        st.pyplot(fig)
    fit_select(
        base_name="null importance",
        delltype=DellType.NULL,
        default_range_start=0.0,
        default_range_end=0.5,
        default_each_calc_size=10)

    ############################
    # feature importance columns
    ############################
    st.write("---")
    st.header("feature importance")
    if st.button("create feature importance"):
        # Calculate feature importance
        fs.create_feature_importance_df(df=src_df)
        save_ifs(fs)
    if fs.feature_importance_p_df is not None:
        fig = plt.figure(figsize=(5, 2))
        plt.bar(fs.feature_importance_p_df["cols"], fs.feature_importance_p_df["mean_importance_p"])
        plt.xticks(rotation=90)
        st.pyplot(fig)
    fit_select(
        base_name="feature importance",
        delltype=DellType.FEATURE_IMPORTANCE,
        default_range_start=0.0,
        default_range_end=0.5,
        default_each_calc_size=10)
    return fs