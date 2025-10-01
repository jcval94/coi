
import pandas as pd
from ..interpret.person import person_interpretation
def build_person_monthly(df: pd.DataFrame) -> pd.DataFrame:
    em = (df.groupby(["month_id","persona_1"], observed=True)
             .agg(n_tx_emit=("monto","count"), sum_emit=("monto","sum"), avg_emit=("monto","mean"),
                  risk_avg_em=("risk_score","mean"), risk_max_emit=("risk_score","max"))
             .rename_axis(index={"persona_1":"persona"}).reset_index())
    re = (df.groupby(["month_id","persona_2"], observed=True)
             .agg(n_tx_recv=("monto","count"), sum_recv=("monto","sum"), avg_recv=("monto","mean"),
                  risk_avg_re=("risk_score","mean"), risk_max_recv=("risk_score","max"))
             .rename_axis(index={"persona_2":"persona"}).reset_index())
    people = pd.merge(em, re, on=["month_id","persona"], how="outer").fillna(
        {"n_tx_emit":0,"sum_emit":0.0,"avg_emit":0.0,"risk_avg_em":0.0,"risk_max_emit":0.0,
         "n_tx_recv":0,"sum_recv":0.0,"avg_recv":0.0,"risk_avg_re":0.0,"risk_max_recv":0.0}
    )
    people["risk_avg_person"] = (people["risk_avg_em"] + people["risk_avg_re"]) / 2.0
    topc = (df[df["nlp_concepto_sospechoso"]!=""]
            .groupby(["month_id","persona_1","nlp_concepto_sospechoso"], observed=True)["monto"]
            .count().reset_index(name="cnt"))
    topc = topc.sort_values(["month_id","persona_1","cnt"], ascending=[True,True,False]).groupby(["month_id","persona_1"]).head(3)
    tops = topc.groupby(["month_id","persona_1"])["nlp_concepto_sospechoso"].apply(list).rename("top_conceptos")
    people = people.merge(tops, left_on=["month_id","persona"], right_on=["month_id","persona_1"], how="left").drop(columns=["persona_1"], errors="ignore")
    people["top_conceptos"] = people["top_conceptos"].apply(lambda x: x if isinstance(x,list) else [])
    people["interp_person"] = people.apply(person_interpretation, axis=1)
    return people.sort_values(["risk_avg_person","sum_emit","sum_recv"], ascending=[False,False,False])
