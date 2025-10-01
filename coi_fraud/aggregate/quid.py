from typing import Tuple

import pandas as pd

from ..config import P
from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_SENDER_ID


def _empty_cases() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fecha_hora_ts",
            COL_SENDER_ID,
            COL_RECEIVER_ID,
            COL_AMOUNT,
            "relacion",
            "feat_quid_rel_label",
            "feat_quid_has_approval",
            "feat_quid_has_comp",
            "feat_quid_value_vs_load_days",
            "feat_quid_score",
            "feat_quid_pair_key",
            "feat_quid_desc_norm",
            "descripcion",
            "transaction_desc",
            "reference_number_trans_desc",
            "risk_score",
        ]
    )


def _empty_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "quid_pair_clave",
            "quid_pair_label",
            "quid_tx_count",
            "quid_score_max",
            "quid_score_avg",
            "quid_manager_ratio",
            "quid_aprob_ratio",
            "quid_comp_ratio",
        ]
    )


def build_quid_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    min_score = float(P.quid_min_score)
    mask = df.get("feat_quid_score", 0) >= min_score
    cases = df.loc[mask].copy()
    if cases.empty:
        return _empty_cases(), _empty_pairs()

    defaults = {
        "feat_quid_has_approval": False,
        "feat_quid_has_comp": False,
        "feat_quid_value_vs_load_days": pd.NA,
        "feat_quid_pair_key": "",
        "feat_quid_desc_norm": "",
        "feat_quid_rel_label": "Par/Indefinido",
    }
    for col, default in defaults.items():
        if col not in cases:
            cases[col] = default

    cols = [
        "fecha_hora_ts",
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        COL_AMOUNT,
        "relacion",
        "feat_quid_rel_label",
        "feat_quid_has_approval",
        "feat_quid_has_comp",
        "feat_quid_value_vs_load_days",
        "feat_quid_score",
        "feat_quid_pair_key",
        "feat_quid_desc_norm",
        "descripcion",
        "transaction_desc",
        "reference_number_trans_desc",
        "risk_score",
    ]
    keep = [c for c in cols if c in cases.columns]
    cases = cases[keep].sort_values(
        ["feat_quid_score", "fecha_hora_ts"], ascending=[False, True]
    )

    pairs = cases[cases["feat_quid_pair_key"].astype(str) != ""].copy()
    if pairs.empty:
        return cases, _empty_pairs()

    pairs["_is_mgr"] = (
        pairs["feat_quid_rel_label"].astype(str).str.contains("Manager", case=False)
    )
    pairs["_has_apr"] = pairs["feat_quid_has_approval"].astype(bool)
    pairs["_has_comp"] = pairs["feat_quid_has_comp"].astype(bool)

    agg = (
        pairs.groupby("feat_quid_pair_key", observed=True)
        .agg(
            quid_tx_count=("feat_quid_score", "count"),
            quid_score_max=("feat_quid_score", "max"),
            quid_score_avg=("feat_quid_score", "mean"),
            quid_manager_ratio=("_is_mgr", "mean"),
            quid_aprob_ratio=("_has_apr", "mean"),
            quid_comp_ratio=("_has_comp", "mean"),
        )
        .reset_index()
        .rename(columns={"feat_quid_pair_key": "quid_pair_clave"})
    )

    def _pair_label(key: str) -> str:
        if not isinstance(key, str) or "|" not in key:
            return key
        a, b = key.split("|", 1)
        return f"{a}↔{b}"

    agg["quid_pair_label"] = agg["quid_pair_clave"].apply(_pair_label)
    agg = agg.sort_values(
        ["quid_score_max", "quid_tx_count"], ascending=[False, False]
    )
    return cases, agg.reset_index(drop=True)
