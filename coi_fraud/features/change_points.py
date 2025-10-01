from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_SENDER_ID


@dataclass
class ChangePointConfig:
    monthly_amount_ratio: float = 4.0
    monthly_count_ratio: float = 3.0
    min_gap_months: int = 3
    min_history_months: int = 2


class ChangePointDetector:
    def __init__(self, cfg: ChangePointConfig):
        self.cfg = cfg

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df["feat_pair_month_amount_ratio"] = 0.0
            df["feat_pair_month_count_ratio"] = 0.0
            df["feat_pair_months_since_prev"] = np.nan
            df["sig_pair_change_point"] = False
            df["sig_pair_new_edge"] = False
            return df

        required = {COL_SENDER_ID, COL_RECEIVER_ID, "month_id", COL_AMOUNT}
        if not required.issubset(df.columns):
            missing = required.difference(df.columns)
            raise ValueError(f"Faltan columnas para ChangePointDetector: {missing}")

        work = df[[COL_SENDER_ID, COL_RECEIVER_ID, "month_id", COL_AMOUNT]].copy()
        work["month_period"] = pd.PeriodIndex(work["month_id"], freq="M")

        group_cols = [COL_SENDER_ID, COL_RECEIVER_ID, "month_period"]
        monthly = (
            work.groupby(group_cols, observed=True)
            .agg(
                month_amount=(COL_AMOUNT, "sum"),
                month_count=(COL_AMOUNT, "count"),
            )
            .reset_index()
        )

        if monthly.empty:
            df["feat_pair_month_amount_ratio"] = 0.0
            df["feat_pair_month_count_ratio"] = 0.0
            df["feat_pair_months_since_prev"] = np.nan
            df["sig_pair_change_point"] = False
            df["sig_pair_new_edge"] = False
            return df

        monthly = monthly.sort_values(group_cols)
        monthly["month_id"] = monthly["month_period"].astype(str)

        pair_cols = [COL_SENDER_ID, COL_RECEIVER_ID]
        prev_amount = monthly.groupby(pair_cols)["month_amount"].shift(1)
        prev_count = monthly.groupby(pair_cols)["month_count"].shift(1)

        prev_period = monthly.groupby(pair_cols)["month_period"].shift(1)
        current_ord = monthly["month_period"].array.asi8.astype("float64")
        prev_ord = prev_period.array.asi8.astype("float64")
        prev_ord[pd.isna(prev_period)] = np.nan
        months_since_prev = current_ord - prev_ord

        global_start = float(monthly["month_period"].min().ordinal)
        months_from_start = current_ord - global_start
        first_mask = prev_amount.isna()
        months_since_prev[first_mask] = months_from_start[first_mask]

        monthly["feat_pair_month_amount_ratio"] = _safe_ratio(
            monthly["month_amount"], prev_amount
        )
        monthly["feat_pair_month_count_ratio"] = _safe_ratio(
            monthly["month_count"], prev_count
        )
        monthly["feat_pair_months_since_prev"] = months_since_prev

        cfg = self.cfg
        change_amount = (
            (prev_amount.fillna(0.0) > 0)
            & (
                monthly["feat_pair_month_amount_ratio"]
                >= float(cfg.monthly_amount_ratio)
            )
        )
        change_count = (
            (prev_count.fillna(0.0) > 0)
            & (
                monthly["feat_pair_month_count_ratio"]
                >= float(cfg.monthly_count_ratio)
            )
        )
        monthly["sig_pair_change_point"] = (change_amount | change_count).astype(bool)

        gap_ok = (
            monthly["feat_pair_months_since_prev"].fillna(np.inf)
            >= float(cfg.min_gap_months)
        )
        history_ok = months_from_start >= float(cfg.min_history_months)
        monthly["sig_pair_new_edge"] = (
            first_mask
            & (monthly["month_amount"] > 0)
            & gap_ok
            & history_ok
        )

        merged = df.merge(
            monthly[
                [
                    *pair_cols,
                    "month_id",
                    "feat_pair_month_amount_ratio",
                    "feat_pair_month_count_ratio",
                    "feat_pair_months_since_prev",
                    "sig_pair_change_point",
                    "sig_pair_new_edge",
                ]
            ],
            how="left",
            on=[COL_SENDER_ID, COL_RECEIVER_ID, "month_id"],
        )

        merged["feat_pair_month_amount_ratio"] = merged[
            "feat_pair_month_amount_ratio"
        ].fillna(0.0)
        merged["feat_pair_month_count_ratio"] = merged[
            "feat_pair_month_count_ratio"
        ].fillna(0.0)
        merged["feat_pair_months_since_prev"] = merged[
            "feat_pair_months_since_prev"
        ]
        merged["sig_pair_change_point"] = merged[
            "sig_pair_change_point"
        ].fillna(False)
        merged["sig_pair_new_edge"] = merged["sig_pair_new_edge"].fillna(False)
        return merged


def _safe_ratio(curr: pd.Series, prev: pd.Series) -> pd.Series:
    prev_filled = prev.fillna(0.0)
    ratio = pd.Series(0.0, index=curr.index, dtype="float64")
    mask = prev_filled > 0
    ratio.loc[mask] = curr.loc[mask] / prev_filled.loc[mask]
    return ratio.replace([np.inf, -np.inf], 0.0)
