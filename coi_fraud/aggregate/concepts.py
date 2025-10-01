import pandas as pd

from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_SENDER_ID


def build_concept_tables(df: pd.DataFrame):
    tx = df.copy()
    txc = tx[tx["nlp_concepto_sospechoso"] != ""].copy()
    agg_concepto = (
        txc.groupby(["month_id", "nlp_concepto_sospechoso"], observed=True)
        .agg(
            tx_count=(COL_AMOUNT, "count"),
            tx_sum=(COL_AMOUNT, "sum"),
            emisores_unicos=(COL_SENDER_ID, "nunique"),
            receptores_unicos=(COL_RECEIVER_ID, "nunique"),
            risk_avg=("risk_score", "mean"),
            risk_p95=(
                "risk_score",
                lambda s: float(s.quantile(0.95)) if len(s) else 0.0,
            ),
        )
        .reset_index()
        .sort_values(["month_id", "risk_p95", "tx_count"], ascending=[True, False, False])
    )
    agg_persona_concepto = (
        txc.groupby(["month_id", COL_SENDER_ID, "nlp_concepto_sospechoso"], observed=True)
        .agg(tx_count=(COL_AMOUNT, "count"), tx_sum=(COL_AMOUNT, "sum"), risk_avg=("risk_score", "mean"))
        .reset_index()
        .rename(columns={COL_SENDER_ID: "persona"})
    )
    txc["pair"] = txc[COL_SENDER_ID].astype(str) + "→" + txc[COL_RECEIVER_ID].astype(str)
    agg_par_concepto = (
        txc.groupby(["month_id", "pair", "nlp_concepto_sospechoso"], observed=True)
        .agg(tx_count=(COL_AMOUNT, "count"), tx_sum=(COL_AMOUNT, "sum"), risk_avg=("risk_score", "mean"))
        .reset_index()
    )
    return agg_concepto, agg_persona_concepto, agg_par_concepto
