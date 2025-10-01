
import pandas as pd
from ..interpret.pair import pair_interpretation
def build_pair_monthly(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["pair"] = tmp["persona_1"].astype(str) + "→" + tmp["persona_2"].astype(str)
    agg = (tmp.groupby(["month_id","pair"], observed=True)
        .agg(tx_count=("monto","count"),
             tx_sum=("monto","sum"),
             risk_max=("risk_score","max"),
             risk_avg=("risk_score","mean"),
             pct_yoyo=("sig_yoyo","mean"),
             pct_smurf=("sig_smurf","mean"),
             pct_freq=("sig_freq","mean"),
             pct_recurrent=("sig_recurrent","mean"))
        .reset_index())
    topc = (tmp[tmp["nlp_concepto_sospechoso"]!=""]
            .groupby(["month_id","pair","nlp_concepto_sospechoso"], observed=True)["monto"]
            .count().reset_index(name="cnt"))
    if len(topc):
        topc = topc.sort_values(["month_id","pair","cnt"], ascending=[True,True,False]).groupby(["month_id","pair"]).head(3)
        tops = topc.groupby(["month_id","pair"])["nlp_concepto_sospechoso"].apply(list).rename("top_conceptos")
        agg = agg.merge(tops, on=["month_id","pair"], how="left")
    else:
        agg["top_conceptos"] = [[] for _ in range(len(agg))]
    agg["interp_pair"] = agg.apply(pair_interpretation, axis=1)
    return agg.sort_values(["risk_max","tx_sum"], ascending=[False,False])
