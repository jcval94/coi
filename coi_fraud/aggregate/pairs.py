import pandas as pd

from ..interpret.pair import pair_interpretation
from ..schemas import COL_AMOUNT, COL_SENDER_ID


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
                "interp_pair",
                "nlp_par_total_transacciones_sospechosas",
                "nlp_par_conceptos_sospechosos_unicos",
                "nlp_par_top_conceptos",
            ]
        )

    tmp = df.copy()
    tmp["pair"] = tmp[COL_SENDER_ID].astype(str) + "→" + tmp[COL_RECEIVER_ID].astype(str)
    for col in ["sig_quid_pro_quo", "sig_reference_reuse"]:
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

    agg["interp_pair"] = agg.apply(pair_interpretation, axis=1)
    agg = agg.sort_values(["risk_max", "tx_sum"], ascending=[False, False])
    agg["top_conceptos"] = agg["nlp_par_top_conceptos"]
    return agg
