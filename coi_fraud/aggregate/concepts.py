
import pandas as pd
def build_concept_tables(df: pd.DataFrame):
    tx = df.copy()
    txc = tx[tx["nlp_concepto_sospechoso"]!=""].copy()
    agg_concepto = (txc.groupby(["month_id","nlp_concepto_sospechoso"], observed=True)
        .agg(tx_count=("monto","count"),
             tx_sum=("monto","sum"),
             emisores_unicos=("persona_1","nunique"),
             receptores_unicos=("persona_2","nunique"),
             risk_avg=("risk_score","mean"),
             risk_p95=("risk_score", lambda s: float(s.quantile(0.95)) if len(s) else 0.0))
        .reset_index()
        .sort_values(["month_id","risk_p95","tx_count"], ascending=[True,False,False]))
    agg_persona_concepto = (txc.groupby(["month_id","persona_1","nlp_concepto_sospechoso"], observed=True)
        .agg(tx_count=("monto","count"), tx_sum=("monto","sum"), risk_avg=("risk_score","mean"))
        .reset_index().rename(columns={"persona_1":"persona"}))
    txc["pair"] = txc["persona_1"].astype(str)+"→"+txc["persona_2"].astype(str)
    agg_par_concepto = (txc.groupby(["month_id","pair","nlp_concepto_sospechoso"], observed=True)
        .agg(tx_count=("monto","count"), tx_sum=("monto","sum"), risk_avg=("risk_score","mean"))
        .reset_index())
    return agg_concepto, agg_persona_concepto, agg_par_concepto
