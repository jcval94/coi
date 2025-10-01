import pandas as pd

from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_SENDER_ID


def _suspicious_concepts(df: pd.DataFrame) -> pd.DataFrame:
    conceptos = df.get("nlp_concepto_sospechoso")
    if conceptos is None:
        return df.iloc[0:0].copy()
    conceptos = conceptos.fillna("").astype("string").str.strip()
    mask = conceptos != ""
    if not mask.any():
        return df.iloc[0:0].copy()
    out = df.loc[mask].copy()
    out["nlp_concepto_sospechoso"] = (
        out["nlp_concepto_sospechoso"].astype("string").str.strip()
    )
    return out


def build_concept_tables(df: pd.DataFrame):
    txc = _suspicious_concepts(df)
    if txc.empty:
        conceptos = pd.DataFrame(
            columns=[
                "nlp_concepto_sospechoso",
                "nlp_concepto_tx_total",
                "nlp_concepto_monto_total",
                "nlp_concepto_emisores_unicos",
                "nlp_concepto_receptores_unicos",
                "nlp_concepto_riesgo_promedio",
                "nlp_concepto_riesgo_p95",
            ]
        )
        persona = pd.DataFrame(
            columns=[
                "persona",
                "nlp_concepto_sospechoso",
                "nlp_persona_concepto_tx_total",
                "nlp_persona_concepto_monto_total",
                "nlp_persona_concepto_riesgo_promedio",
            ]
        )
        par = pd.DataFrame(
            columns=[
                "pair",
                "nlp_concepto_sospechoso",
                "nlp_par_concepto_tx_total",
                "nlp_par_concepto_monto_total",
                "nlp_par_concepto_riesgo_promedio",
            ]
        )
        return conceptos, persona, par

    agg_concepto = (
        txc.groupby("nlp_concepto_sospechoso", observed=True)
        .agg(
            nlp_concepto_tx_total=(COL_AMOUNT, "count"),
            nlp_concepto_monto_total=(COL_AMOUNT, "sum"),
            nlp_concepto_emisores_unicos=(COL_SENDER_ID, "nunique"),
            nlp_concepto_receptores_unicos=(COL_RECEIVER_ID, "nunique"),
            nlp_concepto_riesgo_promedio=("risk_score", "mean"),
            nlp_concepto_riesgo_p95=(
                "risk_score",
                lambda s: float(s.quantile(0.95)) if len(s) else 0.0,
            ),
        )
        .reset_index()
        .sort_values(
            ["nlp_concepto_riesgo_p95", "nlp_concepto_tx_total"],
            ascending=[False, False],
        )
    )

    agg_persona_concepto = (
        txc.groupby([COL_SENDER_ID, "nlp_concepto_sospechoso"], observed=True)
        .agg(
            nlp_persona_concepto_tx_total=(COL_AMOUNT, "count"),
            nlp_persona_concepto_monto_total=(COL_AMOUNT, "sum"),
            nlp_persona_concepto_riesgo_promedio=("risk_score", "mean"),
        )
        .reset_index()
        .rename(columns={COL_SENDER_ID: "persona"})
    )

    txc["pair"] = txc[COL_SENDER_ID].astype(str) + "→" + txc[COL_RECEIVER_ID].astype(str)
    agg_par_concepto = (
        txc.groupby(["pair", "nlp_concepto_sospechoso"], observed=True)
        .agg(
            nlp_par_concepto_tx_total=(COL_AMOUNT, "count"),
            nlp_par_concepto_monto_total=(COL_AMOUNT, "sum"),
            nlp_par_concepto_riesgo_promedio=("risk_score", "mean"),
        )
        .reset_index()
    )
    return agg_concepto, agg_persona_concepto, agg_par_concepto
