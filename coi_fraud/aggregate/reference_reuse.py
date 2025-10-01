from typing import List, Tuple

import numpy as np
import pandas as pd

from ..config import P
from ..schemas import COL_RECEIVER_ID, COL_SENDER_ID


def _pair_key(df: pd.DataFrame) -> pd.Series:
    send = df[COL_SENDER_ID].astype(str)
    recv = df[COL_RECEIVER_ID].astype(str)
    key_ud = send + "|" + recv
    key_du = recv + "|" + send
    return np.where(key_ud < key_du, key_ud, key_du)


def _pair_label(key: str) -> str:
    if not isinstance(key, str) or "|" not in key:
        return key
    a, b = key.split("|", 1)
    return f"{a}↔{b}"


def _empty_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "reference_norm",
            "reference_len",
            "first_ts",
            "last_ts",
            "days_range",
            "n_pairs",
            "pairs",
            "tx_count",
        ]
    )


def _empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fecha_hora_ts",
            COL_SENDER_ID,
            COL_RECEIVER_ID,
            "reference_number_trans_desc",
            "feat_reference_norm",
            "feat_reference_len",
            "sig_reference_reuse",
            "risk_score",
        ]
    )


def build_reference_reuse_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    window_days = int(P.ref_reuse_window_days)
    ref_col = "feat_reference_norm"
    if ref_col not in df.columns:
        return _empty_summary(), _empty_transactions()

    work = df.loc[df[ref_col].astype(str) != ""].copy()
    if work.empty:
        return _empty_summary(), _empty_transactions()

    work["pair_undirected"] = _pair_key(work)
    work["fecha_hora_ts"] = pd.to_datetime(work.get("fecha_hora_ts"), errors="coerce", utc=True)

    grp = work.groupby(ref_col, observed=True)
    first_ts = grp["fecha_hora_ts"].min()
    last_ts = grp["fecha_hora_ts"].max()
    n_pairs = grp["pair_undirected"].nunique()
    tx_count = grp.size()

    def _collect_pairs(series: pd.Series) -> List[str]:
        unique = sorted(set(series.astype(str)))
        return [_pair_label(key) for key in unique]

    pairs = grp["pair_undirected"].apply(_collect_pairs)

    ref_len = (
        work.groupby(ref_col)["feat_reference_len"].max().reindex(first_ts.index)
    )
    summary = pd.DataFrame(
        {
            "reference_norm": first_ts.index,
            "reference_len": ref_len.values,
            "first_ts": first_ts.values,
            "last_ts": last_ts.values,
            "days_range": (last_ts - first_ts).dt.total_seconds().div(86400).values,
            "n_pairs": n_pairs.values,
            "pairs": pairs.values,
            "tx_count": tx_count.values,
        }
    )
    summary["days_range"] = summary["days_range"].replace({np.nan: np.inf})
    summary = summary[
        (summary["n_pairs"] > 1)
        & (summary["days_range"] <= float(window_days))
    ].copy()
    summary = summary.sort_values(
        ["n_pairs", "days_range", "tx_count"], ascending=[False, True, False]
    )

    tx = work.loc[work[ref_col].isin(summary["reference_norm"])].copy()
    if tx.empty:
        tx_table = _empty_transactions()
    else:
        cols = [
            "fecha_hora_ts",
            COL_SENDER_ID,
            COL_RECEIVER_ID,
            "reference_number_trans_desc",
            "feat_reference_norm",
            "feat_reference_len",
            "sig_reference_reuse",
            "risk_score",
        ]
        keep = [c for c in cols if c in tx.columns]
        tx_table = tx[keep].sort_values(
            ["feat_reference_norm", "fecha_hora_ts"], ascending=[True, True]
        )

    return summary.reset_index(drop=True), tx_table.reset_index(drop=True)
