# %%
import pandas as pd
import numpy as np
import polars as pl

from tqdm import tqdm

# %%
def get_num_cols(df: pd.DataFrame)->list[str]:
    num_cols = df.select_dtypes(include='number').columns
    return num_cols

def get_notnum_cols(df: pd.DataFrame)->list[str]:
    num_cols = get_num_cols(df=df)
    not_num_cols = [c for c in df.columns if c not in num_cols]
    return not_num_cols

def _cramersV_pl(df: pl.DataFrame, col1:str, col2: str)->float:
    pivot_table = df.pivot(
        index=col1,
        on=col2,
        values=col1,
        aggregate_function="len").fill_null(0).drop(col1).to_numpy()
    if df[col1].n_unique() == 1 or df[col2].n_unique() == 1:
        return np.nan
    n = len(df)
    expect = np.outer(pivot_table.sum(axis=1), pivot_table.sum(axis=0)) / n #期待度数
    chisq = np.sum((pivot_table - expect) ** 2 / expect) #カイ二乗値
    cor = np.sqrt(chisq / (n * (min(pivot_table.shape) -1))) #クラメール連関係数
    return cor

def _create_cramers_corr_pl(df: pd.DataFrame, verbose=False) -> pd.DataFrame:
    cols = df.columns
    df_corr = pd.DataFrame(np.ones((len(cols), len(cols))))
    df_corr.columns = cols
    df_corr.index = cols
    df_pl = pl.from_pandas(df)
    vtqdm = tqdm if verbose else lambda x: x
    for ci1 in vtqdm(range(0, len(cols)-1)):
        for ci2 in range(ci1+1, len(cols)):
            cor = _cramersV_pl(df=df_pl, col1=cols[ci1], col2=cols[ci2])
            df_corr.iloc[ci1, ci2] = df_corr.iloc[ci2, ci1] = cor
    return df_corr

def _numpy_corr(df: pd.DataFrame)->pd.DataFrame:
    cor = np.corrcoef(df.values.T)
    cor = pd.DataFrame(cor)
    cor.columns = df.columns
    cor.index = df.columns
    return cor

def _calc_correlation_ratio(categories, measurements):
        fcat, _ = pd.factorize(categories)
        cat_num = np.max(fcat)+1
        y_avg_array = np.zeros(cat_num)
        n_array = np.zeros(cat_num)
        for i in range(0,cat_num):
            cat_measures = measurements[np.argwhere(fcat == i).flatten()]
            n_array[i] = len(cat_measures)
            y_avg_array[i] = np.average(cat_measures)
        y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
        numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
        denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
        if numerator == 0:
            eta = 0.0
        else:
            eta = numerator/denominator
        return eta

def _create_full_corr_df(
        df: pd.DataFrame,
        is_num_corr: bool=True,
        is_cat_corr: bool=True,
        is_numcat_corr: bool=True,
        verbose: bool=False
        )->pd.DataFrame:
    cols = df.columns
    cor_df = pd.DataFrame(np.full((len(cols), len(cols)), np.nan))
    cor_df.columns = cols
    cor_df.index = cols

    # 数値列の相関を求める
    num_cols = get_num_cols(df)
    if is_num_corr:
        if verbose:
            print("calclating num x num corr...")
        num_cor_df = _numpy_corr(df=df[num_cols])
        if verbose:
            print("done")
        cor_df.loc[num_cols, num_cols] = num_cor_df.values
    # カテゴリ列の相関を求める
    cat_cols = get_notnum_cols(df)
    if is_cat_corr:
        if verbose:
            print("calclating category x category corr...")
        cat_cor_df = _create_cramers_corr_pl(df=df[cat_cols],verbose=verbose)
        if verbose:
            print("done")
        cor_df.loc[cat_cols, cat_cols] = cat_cor_df.values

    # 数値列とカテゴリ列の相関を求める
    if is_numcat_corr:
        if verbose:
            print("calclating category num x category corr...")
        vtqdm = tqdm if verbose else lambda x: x
        for num_col in vtqdm(num_cols):
            for cat_col in cat_cols:
                cor = _calc_correlation_ratio(categories=df[cat_col].values, measurements=df[num_col].values)
                cor_df.loc[cat_col, num_col] = cor_df.loc[num_col, cat_col] = cor
        if verbose:
            print("done")
    return cor_df

def _get_corr_column_for_drop(df_corr: pd.DataFrame, threshold: float, verbose: bool=False)->tuple[list[str], pd.DataFrame]:
    """
    Delete columns that have high correlation

    Args:
        df_corr (pd.DataFrame): correlation matrix
        threshold (float): Threshold. Columns with correlation above this value will be deleted.
    Returns:
        list[str]: Names of deleted columns
        pd.DataFrame: Detailed information of deleted columns
    """
    org_cols = df_corr.columns

    df_corr = abs(df_corr)
    columns = df_corr.columns

    # 対角線の値を0にする
    for i in range(0, len(columns)):
        df_corr.iloc[i, i] = 0
    detail_info_df = []
    while True:
        columns = df_corr.columns
        max_corr = 0.0
        query_column = None
        target_column = None

        df_max_column_value = df_corr.max()
        max_corr = df_max_column_value.max()
        query_column = df_max_column_value.idxmax()
        target_column = df_corr[query_column].idxmax()

        if max_corr <= threshold:
            # しきい値を超えるものがなかったため終了
            break
        else:
            # しきい値を超えるものがあった場合
            delete_column = None
            saved_column = None

            # その他との相関の絶対値が大きい方を除去
            if np.nanmean(df_corr[query_column].values) <= np.nanmean(df_corr[target_column].values):
                delete_column = target_column
                saved_column = query_column
            else:
                delete_column = query_column
                saved_column = target_column
            detail_info_df.append({
                "delete_column": delete_column,
                "saved_column": saved_column,
                "corr": max_corr}
            )
            # 除去すべき特徴を相関行列から消す（行、列）
            df_corr.drop([delete_column], axis=0, inplace=True)
            df_corr.drop([delete_column], axis=1, inplace=True)
    droped_cols = [c for c in org_cols if c not in df_corr.columns]
    detail_info_df = pd.DataFrame(detail_info_df)
    return droped_cols, detail_info_df

def get_corr_column_for_drop(
        df: pd.DataFrame,
        is_num_corr: bool=True,
        is_cat_corr: bool=True,
        is_numcat_corr: bool=True,
        threshold: float=0.95,
        verbose: bool=False)->tuple[list[str], pd.DataFrame]:
    """
    Delete columns that have high correlation
    Args:
        df (pd.DataFrame): DataFrame
        is_num_corr (bool): Calculate correlation for numerical columns. Default: True
        is_cat_corr (bool): Calculate correlation for categorical columns. Default: True
        is_numcat_corr (bool): Calculate correlation between numerical and categorical columns. Default: True
        threshold (float): Threshold. Default: 0.95
        verbose (bool): Display progress. Default: False
    Returns:
        list[str]: Names of deleted columns
        pd.DataFrame: Detailed information of deleted columns
    """
    # """
    # 相関が高い列を削除する
    # Args:
    #     df (pd.DataFrame): データフレーム
    #     is_num_corr (bool): 数値列の相関を計算するか.Default: True
    #     is_cat_corr (bool): カテゴリ列の相関を計算するか.Default: True
    #     is_numcat_corr (bool): 数値列とカテゴリ列の相関を計算するか.Default: True
    #     threshold (float): しきい値.Default: 0.95
    #     verbose (bool): 進捗を表示するか.Default: False
    # Returns:
    #     list[str]: 削除した列名
    #     pd.DataFrame: 削除した詳細情報
    # """
    df_corr = _create_full_corr_df(
        df=df,
        is_num_corr=is_num_corr,
        is_cat_corr=is_cat_corr,
        is_numcat_corr=is_numcat_corr,
        verbose=verbose)
    droped_cols, detail_info_df = _get_corr_column_for_drop(df_corr=df_corr, threshold=threshold, verbose=verbose)
    return droped_cols, detail_info_df
# %%
if __name__ == "__main__":
    df = pd.DataFrame({
        "a": ["a", "a", "c", "a", "b", "c", "b"],
        "b": ["x", "x", "z", "x", "y", "z", "y"],
        "c": ["x", "y", "x", "x", "y", "y", "x"],
        "d": [1  ,  1 ,  3 ,  1 , 2  ,  3 ,  2],
        "e": [4  ,  1 ,  3 ,  1 , 2  ,  3 ,  4],
    })
    df_corr = _create_full_corr_df(df=df, is_numcat_corr=True)
    display(df_corr)
    droped_cols, detail_info_df = _get_corr_column_for_drop(df_corr=df_corr, threshold=0.5)
    detail_info_df
    display(detail_info_df)