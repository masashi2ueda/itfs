# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy

from enum import Enum
from typing import Callable, Literal
from scipy import stats

if __name__ == "__main__":
    import sys
    sys.path.append("../")
    from itfs import correlation
else:
    from . import correlation
# %%
MIN_DIFF = 1e-5
class DellType(Enum):
    CONST = 1
    CORR = 2
    NULL = 3
    FEATURE_IMPORTANCE = 4
# %%
def drop_cols_ifexist(df: pd.DataFrame, cols: list[str])->pd.DataFrame:
    drop_cols = [c for c in cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df

class IterativeFeatureSelector:
    def __init__(
            self,
            calc_error_importance: Callable[[pd.DataFrame, np.ndarray], dict[str, any]],
            force_drop_cols: list[str] = [],
            ignore_cols: list[str] = [],
            ycol: str| None = None,
            min_col_size: int = 5):
        """ Create an instance for feature selection
        Args:
            calc_error_importance (Callable[[pd.DataFrame, np.ndarray], dict[str, any]]): Function to calculate score and importance.
                Arguments are (xdf, ys)
                Assuming cross validation, the returned dict is as follows:
                {
                    "oof_error" (float): oof_error, # oof error
                    "cv_errors" (np.ndarray[float]): cv_errors, # errors for each fold in cv
                    "importance_df" (pd.DataFrame): importance_df # DataFrame of importance, index is feature name, each column is importance for each fold
                }
            force_drop_cols (list[str]): Columns to be forcibly dropped
            ignore_cols (list[str]): Columns to be ignored.
                These columns are excluded from feature selection.
            ycol (str): Column name of the target variable
            min_col_size (int): Minimum number of columns. Feature selection is not performed if the number of columns is less than this. Default: 5
        """
        # 入力をコピー
        self.calc_error_importance = calc_error_importance
        self.force_drop_cols = force_drop_cols
        self.ignore_cols = ignore_cols
        self.ycol = ycol
        self.min_col_size = min_col_size

        # エラーのリスト
        self.error_list: list[dict] = []

        # 落とす列の名前
        self.selected_dropcols = {dtp: None for dtp in DellType}
        self.accept_ths = {dtp: None for dtp in DellType}
        self.selected_error = {dtp: None for dtp in DellType}

        # 相関行列
        self.cor_df: pd.DataFrame|None = None

        # null importanceのrate
        self.null_importance_p_df: pd.DataFrame|None = None

        # feature importanceのrate
        self.feature_importance_p_df: pd.DataFrame|None = None


    def get_all_drop_selected_cols(
            self,
            is_all_true: bool = False,
            is_use_delltypes: list[DellType] = [],
            is_drop_ycol: bool = True) -> list[str]:
        """ get all drop selected columns
        Args:
            df (pd.DataFrame): DataFrame
            is_all_true (bool): Whether to drop all columns. Default: False
            is_use_delltypes (list[DellType]): Whether to use DellType. Default: []
            is_drop_ycol (bool): Whether to drop ycol. Default: True
        Returns:
            list[str]: Names of columns to be dropped
        """
        drop_cols = []
        if is_all_true:
            is_use_delltypes = [dtp for dtp in DellType]
            is_drop_ycol = True
        drop_cols += self.force_drop_cols
        drop_cols += self.ignore_cols
        for dtp in is_use_delltypes:
            ta_cols = self.selected_dropcols[dtp]
            if ta_cols is not None:
                drop_cols += ta_cols
        if is_drop_ycol:
            drop_cols += [self.ycol]
        return drop_cols

    def drop_selected_cols(
            self,
            df: pd.DataFrame,
            is_all_true: bool = False,
            is_use_delltypes: list[DellType] = [],
            is_drop_ycol: bool = True) -> pd.DataFrame:
        """ Drop selected columns
        Args:
            df (pd.DataFrame): DataFrame
            is_all_true (bool): Whether to drop all columns. Default: False
            is_use_delltypes (list[DellType]): Whether to use DellType. Default: []
            is_drop_ycol (bool): Whether to drop ycol. Default: True
        Returns:
            pd.DataFrame: DataFrame with selected columns dropped
        """
        drop_cols = self.get_all_drop_selected_cols(
            is_all_true=is_all_true,
            is_use_delltypes=is_use_delltypes,
            is_drop_ycol=is_drop_ycol)

        return drop_cols_ifexist(df=df, cols=drop_cols)

    def _update_error_list(
            self,
            delltype: DellType,
            df: pd.DataFrame,
            ta_df: pd.DataFrame,
            temp_drop_cols: list[str],
            th: float,
            verbose: bool = False) -> None:
        """ エラーリストを更新する
        Args:
            delltype (DellType): Type of deletion
            df (pd.DataFrame): オリジナルのデータフレーム
            ta_df (pd.DataFrame): すでに不要なことが決定している特徴量を除外した後のデータフレーム
            temp_drop_cols (list[str]): 今回削除する列名
            error_dict_list (list[dict]|None): エラーのリスト
            th (float): 登録するしきい値
            verbose (bool): デバッグ情報を表示するかどうか
        """
        # 今回の不要な列を消す
        xdf = ta_df.drop(columns=temp_drop_cols)

        # 予測対象の列
        ys = df[self.ycol]

        # 列が少なすぎる場合は終了
        if xdf.shape[1] < self.min_col_size:
            if verbose:
                print("xdf.shape:", xdf.shape, "too small columns size")
            return

        # 今回使う列名の一覧
        concat_col_name = "__".join(xdf.columns.tolist())
        # scoreを計算するべきかどうか？
        is_calc = True
        # もう何回か計算している　かつ
        if len(self.error_list) != 0:
            # 今回の列名がすでにある　なら計算しない
            if concat_col_name in [item["concat_col_name"] for item in self.error_list]:
                if verbose:
                    print("this concat_col_name is already calculated")
                is_calc = False

        if is_calc:
            # まだないので計算する
            res_dict = self.calc_error_importance(xdf, ys)
            if verbose:
                print(f"calculated error:{res_dict}")
        else:
            # すでにあるので、それを使う
            ex_item = [item for item in self.error_list if item["concat_col_name"] == concat_col_name][0]
            res_dict = copy.deepcopy(ex_item)

        # listに登録するかどうか
        is_regist = True
        # もう何回か登録している かつ
        if len(self.error_list) != 0:
            # 今回のthがすでにある　なら登録しない
            if 1<=len([item for item in self.error_list if item["th"] == th and item["delltype"] == delltype]):
                is_regist = False
                if verbose:
                    print("this th is already registered")

        # 登録する
        if is_regist:
            use_col_names = xdf.columns.tolist()
            res_dict["delltype"] = delltype
            res_dict["th"] = th
            res_dict["temp_drop_cols"] = temp_drop_cols
            res_dict["concat_col_name"] = concat_col_name
            res_dict["use_col_names"] = use_col_names
            self.error_list.append(res_dict)

    def fit_const_cols(self, df: pd.DataFrame) -> list[str]:
        """ Delete columns with constant values.
        The names of the columns with constant values to be deleted are stored in self.const_cols.
        Args:
            df (pd.DataFrame): DataFrame
        Returns:
            list[str]: Names of columns with constant values to be deleted
        Note:
            A constant column is one where all values are the same.
            Calculate accuracy after removing constant columns.
        """
        const_type = DellType.CONST
        # 不要なものを消す
        ta_df = self.drop_selected_cols(df=df)
        # 一定のものを抽出
        const_cols = []
        # 各列のユニークな数をカウント
        for c in ta_df.columns:
            # 全部Noneだと0になる
            if ta_df[c].nunique() <= 1:
                const_cols.append(c)
        self.selected_dropcols[const_type] = const_cols
        # constを抜いて一回計算する
        ta_df = self.drop_selected_cols(df=df, is_use_delltypes=[])
        self._update_error_list(delltype=const_type, df=df, ta_df=ta_df, temp_drop_cols=const_cols,th=None)
        return self.selected_dropcols[const_type]

    def calc_error_p(self, min_item, target_item: dict) -> float:
        min_oof_error = min_item["oof_error"]
        min_cv_errors = min_item["cv_errors"]

        oof_error = target_item["oof_error"]

        min_cv_errors_sd = np.std(min_cv_errors, ddof=1)
        diff_from_min = oof_error - min_oof_error
        diff_from_min_nmd = diff_from_min / min_cv_errors_sd
        return diff_from_min_nmd

    def _get_orless_delltype(self, delltype: DellType) -> list[DellType]:
        return [dtp for dtp in DellType if dtp.value <= delltype.value]
    def _get_lessthan_delltype(self, delltype: DellType) -> list[DellType]:
        return [dtp for dtp in DellType if dtp.value < delltype.value]

    def _get_before_min_error_item(self, delltype: DellType) -> dict:
        # delltypeより前の全てのエラーの中で最小のものを取得
        belo_types = self._get_orless_delltype(delltype=delltype)
        before_error_list = [item for item in self.error_list if item["delltype"] in belo_types]
        if len(before_error_list) == 0:
            return None
        min_error_item = min(before_error_list, key=lambda x: x["oof_error"])
        return min_error_item

    def _errorlist2df(self, delltype: DellType) -> pd.DataFrame:
        # エラーリストをDataFrameに変換
        error_df = pd.DataFrame([item for item in self.error_list if item["delltype"] == delltype])
        # エラーリストを閾値でソート
        error_df = error_df.sort_values("th").reset_index(drop=True)
        # 使う列の数を追加
        error_df["use_col_size"] = [len(cols) for cols in error_df["use_col_names"]]
        # delltypeより前の全てのエラーの中で最小のものを取得
        min_error_item = self._get_before_min_error_item(delltype=delltype)
        # 最小のエラーより各エラーが大きい効果量を計算
        error_ps = []
        for i, row in error_df.iterrows():
            p = self.calc_error_p(min_item=min_error_item, target_item=row)
            error_ps.append(p)
        error_df["error_ps"] = error_ps
        return error_df

    def _plot_state(
            self,
            delltype: DellType,
            each_desc: str,
            ROW: int,
            pi: int,
            is_errorp_log:bool= False)->tuple[int, pd.DataFrame]:
        # errorのdfにする
        error_df = self._errorlist2df(delltype=delltype)
        min_idx = error_df["oof_error"].idxmin()
        min_error_th = error_df["th"].iloc[min_idx]

        # 最小のエラー
        global_min_error = self._get_before_min_error_item(delltype=delltype)["oof_error"]

        # グラフを描画
        c = plt.get_cmap("tab10")

        # 閾値とエラーの様子(cv + 平均)
        def error_plot(is_plot_each_cv: bool):
            c2p = lambda c: (c[0], c[1], c[2], 0.1)
            if is_plot_each_cv:
                for i, row in error_df.iterrows():
                    label = "cv_errors" if i == 0 else None
                    plt.scatter([row["th"]]*len(row["cv_errors"]), row["cv_errors"], color=c2p(c(0)), label=label, marker="x")
            plt.plot(error_df["th"], error_df["oof_error"], color=c(1), label="oof_error", marker="x")
            plt.axvline(min_error_th, color="red", linestyle="--", label="min mean")
            plt.axhline(global_min_error, color="pink", linestyle="--", label="global min error")
            plt.xlabel(each_desc)
            plt.ylabel("error")
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.subplot(ROW, 1, pi)
        plt.title("th vs error(cv + mean)")
        error_plot(is_plot_each_cv=True)
        pi += 1

        # 閾値とエラーの様子(平均のみ)
        plt.subplot(ROW, 1, pi)
        plt.title("th vs error( mean)")
        error_plot(is_plot_each_cv=False)
        pi += 1

        # 閾値とp値の様子
        plt.subplot(ROW, 1, pi)
        plt.title("th vs error_p\n(normalized diff from min error)")
        # y軸をlogにする
        if is_errorp_log:
            use_error_ps = [max(v, MIN_DIFF) for v in error_df["error_ps"]]
            hs = [0.001, 0.01, 0.1, 1]
            hlabel = "error_rate=\n[" + ", ".join([f"{h}" for h in hs]) + "]"
            for hi, h in enumerate(hs):
                label = hlabel if hi == 0 else None
                plt.axhline(h, color=(0, 0, 0, 0.1), linestyle="solid", label=label)
            plt.yscale("log")
        plt.plot(error_df["th"], use_error_ps, marker="x")
        plt.axvline(min_error_th, color="red", linestyle="--", label="minimum error")
        plt.xlabel("th")
        plt.ylabel("error_nmd_diff")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        pi += 1
    
        return pi, error_df

    def _plot_select_core(
            self,
            delltype: DellType,
            accept_th: float,
            is_errorp_log:bool = False):

        # errorのdfにする
        error_df = self._errorlist2df(delltype=delltype)
        min_idx = error_df["oof_error"].idxmin()

        # グラフを描画
        ROW = 1
        fig = plt.figure(figsize=(5, ROW * 3))
        c = plt.get_cmap("tab10")
        plt.subplots_adjust(hspace=1.0)

        pi = 1

        plt.subplot(ROW, 1, pi)
        plt.title("column size vs error_p\n(normalized diff from min error)")
        # y軸をlogにする
        if is_errorp_log:
            use_error_ps = [max(v, MIN_DIFF) for v in error_df["error_ps"]]
            hs = [0.001, 0.01, 0.1, 1]
            hlabel = "error_rate=\n[" + ", ".join([f"{h}" for h in hs]) + "]"
            for hi, h in enumerate(hs):
                label = hlabel if hi == 0 else None
                plt.axhline(h, color=(0, 0, 0, 0.1), linestyle="solid", label=label)
            plt.yscale("log")
        ta_df = error_df[error_df["error_ps"] <= accept_th]
        min_colsize_idx = None
        if ta_df.shape[0] > 0:
            min_colsize_idx = ta_df.index[np.argmin(ta_df["use_col_size"].values)]
        plt.plot(error_df["use_col_size"], use_error_ps, marker="x")
        plt.axvline(error_df["use_col_size"].values[min_idx], color="red", linestyle="--", label="minimum error")
        plt.axhline(accept_th, color="green", linestyle="dashed", label="accept pth from min error")
        if min_colsize_idx is not None:
            plt.axvline(error_df["use_col_size"].values[min_colsize_idx], color="green", linestyle="dotted", label="acceptable minimum column size")
        plt.xlabel("column size")
        plt.ylabel("error_nmd_diff")
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        pi += 1

        # 落とす列を決定
        drop_cols = []
        if min_colsize_idx is not None:
            drop_cols = error_df["temp_drop_cols"].values[min_colsize_idx]
            # 選ばれたものを保存
            self.selected_error = copy.deepcopy(ta_df.loc[min_colsize_idx, :].to_dict())
        return drop_cols, error_df, fig

    def create_cor_df(self, df: pd.DataFrame):
        """ Create a correlation matrix
        Args:
            df (pd.DataFrame): DataFrame
        """
        # 不要なものを消す
        lessthan_delltypes = self._get_lessthan_delltype(delltype=DellType.CORR)
        ta_df = self.drop_selected_cols(df=df,is_use_delltypes=lessthan_delltypes)

        # 相関行列がまだないなら作成
        self.cor_df = correlation._create_full_corr_df(df=ta_df)

    def create_cor_coltable_df(self):
        cols = self.cor_df.columns
        cor_col_df = []
        for r in range(1, len(cols)-1):
            for c in range(r+1, len(cols)):
                cor_col_df.append({
                    "col1": cols[r],
                    "col2": cols[c],
                    "cor": self.cor_df.iloc[r,c]
                })
        cor_col_df = pd.DataFrame(cor_col_df)
        cor_col_df = cor_col_df.sort_values("cor", ascending=False).reset_index(drop=True)
        return cor_col_df

    def calc_error_each_threshold(
            self,
            df: pd.DataFrame,
            delltype: DellType,
            th: float,
            verbose: bool = False):
        """ Delete columns with high correlation for each threshold and calculate error
        Args:
            df (pd.DataFrame): DataFrame
            delltype (DellType): Type of deletion
            th (float): Threshold
            verbose (bool): Whether to display debug information
        """

        # 不要なものを消す
        # delltype未満のものを取得して削除
        lessthan_delltypes = self._get_lessthan_delltype(delltype=delltype)
        ta_df = self.drop_selected_cols(df=df,is_use_delltypes=lessthan_delltypes)

        if verbose:
            print("ta_df.shape:", ta_df.shape)
        if delltype == DellType.CORR:
            # 相関がcor_thより高く、削除すべき列名を取得
            temp_drop_cops, detail_df = correlation._get_corr_column_for_drop(df_corr=self.cor_df, threshold=th)
        elif delltype == DellType.NULL:
            # null importanceのrate
            temp_drop_cops = self.null_importance_p_df[self.null_importance_p_df["p_value"]<=th]["cols"].values.tolist()
        elif delltype == DellType.FEATURE_IMPORTANCE:
            # 落とす列を決定
            temp_drop_cops = self.feature_importance_p_df[th > self.feature_importance_p_df["mean_importance_p"]]["cols"].values.tolist()

        if verbose:
            print("th:", th)
            print("temp_drop_cops:", temp_drop_cops)

        # エラーリストを更新
        self.cor_error_list = self._update_error_list(
            delltype=delltype,
            df=df,
            ta_df=ta_df,
            temp_drop_cols=temp_drop_cops,
            th=th,
            verbose=verbose)


    def plot_state(self, delltype: DellType, is_errorp_log: bool=True)->tuple[plt.Figure, pd.DataFrame]:
        """ Plot the state of correlation
        Args:
            is_errorp_log (bool): Whether to use a log scale for the error rate. Default: True
        Returns:
            plt.Figure: Figure
            pd.DataFrame: Error DataFrame
        """
        fig = plt.figure(figsize=(5, 8))
        plt.subplots_adjust(hspace=1.0)
        ROW = 1 + 3
        pi = 1
        plt.subplot(ROW, 1, pi)

        same_error_list = [item for item in self.error_list if item["delltype"] == delltype]

        if delltype == DellType.CORR:
            each_desc = "th(delete th<=corr columns)"
            plt.title("colleration value")
            cor_vals = self.cor_df.values
            cors = []
            for rw in range(1, cor_vals.shape[0]):
                for cl in range(rw + 1, cor_vals.shape[1]):
                    cors.append(cor_vals[rw, cl])
            cors = np.sort(cors)
            plt.scatter(np.arange(len(cors)), cors, marker = "x", color=(0, 0, 1, 0.1))
            for ei, ed in enumerate(same_error_list):
                label = "tryed th " if ei == 0 else None
                plt.axhline(ed["th"], color=(0, 0, 0, 0.1), label=label)
                plt.axhline(-ed["th"], color=(0, 0, 0, 0.1))
            plt.ylabel("correlation")
            plt.ylim(-1.1, 1.1)

        elif delltype == DellType.NULL:
            # nullの値の様子
            each_desc="th(delete null importance<th columns)"
            plt.title("null importance rate\n(null importandce < importance)")
            vals = self.null_importance_p_df["p_value"].values
            vals = np.sort(vals)
            plt.bar(self.null_importance_p_df["cols"].values, self.null_importance_p_df["p_value"].values, color=(0, 0, 1, 0.3))
            for ei, ed in enumerate(same_error_list):
                label = "th tryed" if ei == 0 else None
                plt.axhline(ed["th"], color=(0, 0, 0, 0.1), label=label)
            plt.ylabel("null importance")
            plt.ylim(-0.1, 1.1)
            plt.xticks(rotation=70)

        elif delltype == DellType.FEATURE_IMPORTANCE:
            # importanceの様子を描画
            each_desc="th(delete th > feature importance columns)"
            plt.title("feature importance\n(delte th>feature importance)")
            plt.bar(
                self.feature_importance_p_df["cols"].values,
                self.feature_importance_p_df["mean_importance_p"].values,
                color=(0, 0, 1, 0.3))
            for ei, ed in enumerate(same_error_list):
                label = "tryed th" if ei == 0 else None
                plt.axhline(ed["th"], color=(0, 0, 0, 0.1), label=label)
            plt.ylabel("mean_importance_p")
            plt.xticks(rotation=90)

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        pi += 1
        pi, error_df = self._plot_state(
            delltype=delltype,
            each_desc=each_desc,
            ROW=ROW, pi=pi, is_errorp_log=is_errorp_log)
        return fig, error_df
        
    def plot_select(
            self,
            delltype: DellType,
            accept_th: float,
            is_errorp_log: bool = True)->tuple[pd.DataFrame, plt.Figure]:

        """ Delete columns with high correlation
        The minimum error group is defined as min errors.
        The error group for each threshold is defined as th errors.
        Calculate how much worse each th errors is compared to min errors using Cohen's d.

        Args:
            delltype (DellType): Type of deletion
            accept_th (float): Threshold
            is_errorp_log (bool): Whether to log the error probability. Default: True
        Returns:
            pd.DataFrame: DataFrame of errors
            plt.Figure: Figure of the plot
        """
        # エラーの閾値を保存
        self.accept_ths[delltype] = accept_th

        # 詳細を描画
        drop_cols, error_df, fig = self._plot_select_core(
            delltype=delltype,
            accept_th=accept_th,
            is_errorp_log=is_errorp_log)

        # 落とす列を決定
        self.selected_dropcols[delltype] = drop_cols

        return error_df, fig

    def _normalize_importance_df(self, importance_df: pd.DataFrame)->pd.DataFrame:
        dst_df = importance_df.copy()
        for c in dst_df.columns:
            dst_df[c] = dst_df[c] / dst_df[c].sum()
        return dst_df

    def _calc_null_importance_p(
            self,
            importance: float,
            null_importances: np.ndarray[float])->float:
        # importance_df = pd.DataFrame(importance_df.mean(axis=1))
        # importance_df.columns = ["mean_importance"]
        # col = importance_df.index[0]
        # # print(col)
        # importance = importance_df.loc[col]["mean_importance"]
        # # print(val)
        # null_importances = null_importance_df.loc[col].values
        # print(null_vals)
        all_importances = np.concatenate([null_importances, [importance]])
        # print(all_importances)
        all_importance_argsorts = all_importances.argsort()
        # print(all_importance_argsorts)
        # importanceは何番目に大きいか？
        importance_idx = np.where(all_importance_argsorts == len(all_importances)-1)[0][0]
        # print(importance_idx)
        importnce_idx_p = importance_idx / (len(all_importances)-1)
        # print(importnce_idx_p)
        return importnce_idx_p

    def create_null_df(
            self, df: pd.DataFrame, 
            null_times: int = 5):
        """ Create a null importance DataFrame
        Args:
            df (pd.DataFrame): DataFrame
            null_times (int): Number of times to calculate null importance. Default: 5
        """
        
        # 不要なものを消す
        lessthan_delltypes = self._get_lessthan_delltype(delltype=DellType.NULL)
        ta_df = self.drop_selected_cols(df=df,is_use_delltypes=lessthan_delltypes)

        xdf = ta_df
        ys = df[self.ycol]

        # 通常のfeature importanceを計算
        res = self.calc_error_importance(xdf=xdf, ys=ys)
        importance_df = res["importance_df"]
        importance_df = self._normalize_importance_df(importance_df)
        # null importanceを計算
        def myshuffle(vals: np.ndarray)->np.ndarray:
            vals = [v for v in vals]
            np.random.shuffle(vals)
            vals = np.array(vals)
            return vals
        null_importance_df = []
        for ni in range(null_times):
            null_res = self.calc_error_importance(xdf=xdf, ys=myshuffle(ys))
            temp_null_importance_df = null_res["importance_df"]
            temp_null_importance_df = self._normalize_importance_df(temp_null_importance_df)
            temp_null_importance_df.columns = [f"{ni}_{c}" for c in temp_null_importance_df.columns]
            null_importance_df.append(temp_null_importance_df)
        null_importance_df = pd.concat(null_importance_df, axis=1)

        # p値を計算
        cols = []
        importance_ps = []
        for row in range(len(importance_df)):
            col = importance_df.index[row]

            importances = importance_df.iloc[row, :].values
            importance = np.mean(importances)

            null_importances = null_importance_df.iloc[row, :].values
        
            importance_p = self._calc_null_importance_p(importance, null_importances)

            cols.append(col)
            importance_ps.append(importance_p)
        null_importance_p_df = pd.DataFrame({"cols": cols, "p_value": importance_ps})
        null_importance_p_df = null_importance_p_df.sort_values("p_value").reset_index(drop=True)
        self.null_importance_p_df = null_importance_p_df

    def create_feature_importance_df(self, df: pd.DataFrame):
        """ Create a feature importance DataFrame
        Args:
            df (pd.DataFrame): DataFrame
        """
        # 不要なものを消す
        # 不要なものを消す
        lessthan_delltypes = self._get_lessthan_delltype(delltype=DellType.FEATURE_IMPORTANCE)
        ta_df = self.drop_selected_cols(df=df,is_use_delltypes=lessthan_delltypes)
        xdf = ta_df
        ys = df[self.ycol]
        # importacneを出す
        res_dict = self.calc_error_importance(xdf=xdf, ys=ys)
        importance_df = res_dict["importance_df"]
        mean_importances = np.mean(importance_df.values, axis=1)
        mean_importance_df = pd.DataFrame()
        # 平均の重要度を計算
        mean_importance_df["mean_importance"] = mean_importances
        mean_importance_df["cols"] = importance_df.index
        # 重要度が高い順にソート
        mean_importance_df = mean_importance_df.sort_values("mean_importance", ascending=False)
        # maxが1になるように正規化
        mean_importance_df["mean_importance_p"] = mean_importance_df["mean_importance"] / mean_importance_df["mean_importance"].max()
        self.feature_importance_p_df = mean_importance_df.reset_index(drop=True)



# %%

# %%

# %%
