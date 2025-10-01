import pandas as pd

from ..interpret.pair import pair_interpretation
from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_RELATION, COL_SENDER_ID


CASE12_MIN_TX = 3
CASE12_OFUSCATED_TOL_RATIO = 0.03


def _compute_case12_tandas(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "pair",
                "caso12_par_tanda_tx_count",
                "caso12_par_tanda_monto_promedio",
                "caso12_par_tanda_no_ofuscada_flag",
                "caso12_par_tanda_ofuscada_flag",
            ]
        )

    teammates = df[df[COL_RELATION] == "companero_equipo"].copy()
    if teammates.empty:
        return pd.DataFrame(
            columns=[
                "pair",
                "caso12_par_tanda_tx_count",
                "caso12_par_tanda_monto_promedio",
                "caso12_par_tanda_no_ofuscada_flag",
                "caso12_par_tanda_ofuscada_flag",
            ]
        )

    teammates["pair"] = (
        teammates[COL_SENDER_ID].astype(str) + "→" + teammates[COL_RECEIVER_ID].astype(str)
    )
    grouped = teammates.groupby("pair", observed=True)[COL_AMOUNT]

    summary = grouped.agg(
        caso12_par_tanda_tx_count="count",
        _unique_amounts=pd.Series.nunique,
        caso12_par_tanda_monto_promedio="mean",
        _min="min",
        _max="max",
    ).reset_index()

    summary["caso12_par_tanda_monto_promedio"] = (
        summary["caso12_par_tanda_monto_promedio"].astype(float).fillna(0.0)
    )
    summary["_range"] = summary["_max"] - summary["_min"]

    summary["caso12_par_tanda_no_ofuscada_flag"] = (
        (summary["caso12_par_tanda_tx_count"] >= CASE12_MIN_TX)
        & (summary["_unique_amounts"] == 1)
    ).astype(int)

    tolerance = (
        summary["caso12_par_tanda_monto_promedio"].abs() * CASE12_OFUSCATED_TOL_RATIO
    ).clip(lower=0.0)
    summary["caso12_par_tanda_ofuscada_flag"] = (
        (summary["caso12_par_tanda_tx_count"] >= CASE12_MIN_TX)
        & (summary["_unique_amounts"] > 1)
        & (summary["_range"].fillna(0.0) <= tolerance.fillna(0.0))
    ).astype(int)

    cols = [
        "pair",
        "caso12_par_tanda_tx_count",
        "caso12_par_tanda_monto_promedio",
        "caso12_par_tanda_no_ofuscada_flag",
        "caso12_par_tanda_ofuscada_flag",
    ]
    return summary[cols]


def build_pair_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "pair",
                "tx_count",
                "tx_sum",
                "risk_max",
                "risk_avg",
                "pct_yoyo",
                "pct_smurf",
                "pct_freq",
                "pct_recurrent",
                "pct_quid_pro_quo",
                "pct_reference_reuse",
                "pct_change_point",
                "pct_new_edge",
                "interp_pair",
                "nlp_par_total_transacciones_sospechosas",
                "nlp_par_conceptos_sospechosos_unicos",
                "nlp_par_top_conceptos",
            ]
        )

    tmp = df.copy()
    tmp["pair"] = tmp[COL_SENDER_ID].astype(str) + "→" + tmp[COL_RECEIVER_ID].astype(str)
    for col in ["sig_quid_pro_quo", "sig_reference_reuse", "sig_pair_change_point", "sig_pair_new_edge"]:
        if col not in tmp:
            tmp[col] = 0.0
    agg = (
        tmp.groupby("pair", observed=True)
        .agg(
            tx_count=(COL_AMOUNT, "count"),
            tx_sum=(COL_AMOUNT, "sum"),
            risk_max=("risk_score", "max"),
            risk_avg=("risk_score", "mean"),
            pct_yoyo=("sig_yoyo", "mean"),
            pct_smurf=("sig_smurf", "mean"),
            pct_freq=("sig_freq", "mean"),
            pct_recurrent=("sig_recurrent", "mean"),
            pct_quid_pro_quo=("sig_quid_pro_quo", "mean"),
            pct_reference_reuse=("sig_reference_reuse", "mean"),
            pct_change_point=("sig_pair_change_point", "mean"),
            pct_new_edge=("sig_pair_new_edge", "mean"),
        )
        .reset_index()
    )

    suspicious_mask = tmp.get("nlp_concepto_sospechoso").fillna("").astype("string").str.strip() != ""
    tmp_suspicious = tmp.loc[suspicious_mask].copy()
    if not tmp_suspicious.empty:
        concept_counts = (
            tmp_suspicious.groupby("pair", observed=True)["nlp_concepto_sospechoso"]
            .agg(
                nlp_par_total_transacciones_sospechosas="count",
                nlp_par_conceptos_sospechosos_unicos="nunique",
            )
            .reset_index()
        )
        topc = (
            tmp_suspicious.groupby(["pair", "nlp_concepto_sospechoso"], observed=True)[COL_AMOUNT]
            .count()
            .reset_index(name="cnt")
        )
        topc = (
            topc.sort_values(["pair", "cnt"], ascending=[True, False])
            .groupby("pair")
            .head(3)
        )
        tops = (
            topc.groupby("pair")["nlp_concepto_sospechoso"].apply(list).rename("nlp_par_top_conceptos")
        )
        agg = agg.merge(concept_counts, on="pair", how="left")
        agg = agg.merge(tops, on="pair", how="left")
    else:
        agg["nlp_par_total_transacciones_sospechosas"] = 0
        agg["nlp_par_conceptos_sospechosos_unicos"] = 0
        agg["nlp_par_top_conceptos"] = [[] for _ in range(len(agg))]

    for col in [
        "nlp_par_total_transacciones_sospechosas",
        "nlp_par_conceptos_sospechosos_unicos",
    ]:
        if col not in agg:
            agg[col] = 0
        else:
            agg[col] = agg[col].fillna(0)
    if "nlp_par_top_conceptos" not in agg:
        agg["nlp_par_top_conceptos"] = [[] for _ in range(len(agg))]
    else:
        agg["nlp_par_top_conceptos"] = agg["nlp_par_top_conceptos"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    case12 = _compute_case12_tandas(df)
    agg = agg.merge(case12, on="pair", how="left")
    for col, fill_value in [
        ("caso12_par_tanda_tx_count", 0),
        ("caso12_par_tanda_monto_promedio", 0.0),
        ("caso12_par_tanda_no_ofuscada_flag", 0),
        ("caso12_par_tanda_ofuscada_flag", 0),
    ]:
        if col not in agg:
            agg[col] = fill_value
        else:
            agg[col] = agg[col].fillna(fill_value)

    agg["interp_pair"] = agg.apply(pair_interpretation, axis=1)
    agg = agg.sort_values(["risk_max", "tx_sum"], ascending=[False, False])
    agg["top_conceptos"] = agg["nlp_par_top_conceptos"]
    return agg
