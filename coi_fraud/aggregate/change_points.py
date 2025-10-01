from typing import Tuple

import numpy as np
import pandas as pd

from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_RELATION, COL_SENDER_ID


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            COL_SENDER_ID,
            COL_RECEIVER_ID,
            "pair",
            "month_id",
            COL_RELATION,
            "sig_pair_change_point",
            "sig_pair_new_edge",
            "feat_pair_month_amount_ratio",
            "feat_pair_month_count_ratio",
            "feat_pair_months_since_prev",
            "month_tx_count",
            "month_tx_sum",
            "risk_score_max",
        ]
    )


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            COL_SENDER_ID,
            COL_RECEIVER_ID,
            "pair",
            "change_point_eventos",
            "new_edge_eventos",
            "risk_score_max",
        ]
    )


def build_change_point_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty or (
        "sig_pair_change_point" not in df.columns
        and "sig_pair_new_edge" not in df.columns
    ):
        return _empty_events(), _empty_pairs()

    work = df.copy()
    for col in ["sig_pair_change_point", "sig_pair_new_edge"]:
        if col not in work:
            work[col] = False

    flagged = work.loc[work["sig_pair_change_point"] | work["sig_pair_new_edge"]].copy()
    if flagged.empty:
        return _empty_events(), _empty_pairs()

    flagged["pair"] = (
        flagged[COL_SENDER_ID].astype(str) + "→" + flagged[COL_RECEIVER_ID].astype(str)
    )

    def _rel_mode(series: pd.Series) -> str:
        counts = series.value_counts(dropna=False)
        if counts.empty:
            return ""
        return counts.idxmax()

    events = (
        flagged.groupby([COL_SENDER_ID, COL_RECEIVER_ID, "pair", "month_id"], observed=True)
        .agg(
            **{
                COL_RELATION: (COL_RELATION, _rel_mode),
                "sig_pair_change_point": ("sig_pair_change_point", "max"),
                "sig_pair_new_edge": ("sig_pair_new_edge", "max"),
                "feat_pair_month_amount_ratio": ("feat_pair_month_amount_ratio", "max"),
                "feat_pair_month_count_ratio": ("feat_pair_month_count_ratio", "max"),
                "feat_pair_months_since_prev": ("feat_pair_months_since_prev", "max"),
                "month_tx_count": (COL_AMOUNT, "count"),
                "month_tx_sum": (COL_AMOUNT, "sum"),
                "risk_score_max": ("risk_score", "max"),
            }
        )
        .reset_index()
    )

    events["feat_pair_month_amount_ratio"] = events[
        "feat_pair_month_amount_ratio"
    ].fillna(0.0)
    events["feat_pair_month_count_ratio"] = events[
        "feat_pair_month_count_ratio"
    ].fillna(0.0)
    events["feat_pair_months_since_prev"] = events[
        "feat_pair_months_since_prev"
    ].replace({np.nan: np.inf})

    pairs = (
        events.groupby([COL_SENDER_ID, COL_RECEIVER_ID, "pair"], observed=True)
        .agg(
            change_point_eventos=("sig_pair_change_point", "sum"),
            new_edge_eventos=("sig_pair_new_edge", "sum"),
            risk_score_max=("risk_score_max", "max"),
        )
        .reset_index()
    )

    events = events.sort_values(
        ["sig_pair_change_point", "sig_pair_new_edge", "risk_score_max"],
        ascending=[False, False, False],
    )
    pairs = pairs.sort_values(
        ["change_point_eventos", "new_edge_eventos", "risk_score_max"],
        ascending=[False, False, False],
    )
    return events, pairs
