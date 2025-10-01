
import pandas as pd
def monthly_percentiles(df, score_col="risk_score", month_col="month_id"):
    g = df.groupby(month_col)[score_col]
    p = g.quantile([0.80,0.90,0.95]).unstack(1).reset_index().rename(columns={0.8:"p80",0.9:"p90",0.95:"p95"})
    cnt = g.count().reset_index().rename(columns={score_col:"n_tx_mes"})
    return p.merge(cnt, on=month_col, how="left")

def add_norm_and_tier(df, calib, score_col="risk_score", month_col="month_id"):
    df = df.merge(calib, on=month_col, how="left")
    df["risk_score_norm"] = df.groupby(month_col)[score_col].rank(pct=True)
    def tier_row(r):
        if r[score_col] >= r.get("p95", float("inf")): return "CRITICO"
        if r[score_col] >= r.get("p90", float("inf")): return "ALTO"
        if r[score_col] >= r.get("p80", float("inf")): return "MEDIO"
        return "BAJO"
    df["risk_tier"] = df.apply(tier_row, axis=1)
    return df
