import pandas as pd

from ..interpret.person import person_interpretation
from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_SENDER_ID


def build_person_monthly(df: pd.DataFrame) -> pd.DataFrame:
    em = (
        df.groupby(["month_id", COL_SENDER_ID], observed=True)
        .agg(
            n_tx_emit=(COL_AMOUNT, "count"),
            sum_emit=(COL_AMOUNT, "sum"),
            avg_emit=(COL_AMOUNT, "mean"),
            risk_avg_em=("risk_score", "mean"),
            risk_max_emit=("risk_score", "max"),
        )
        .rename_axis(index={COL_SENDER_ID: "persona"})
        .reset_index()
    )
    re = (
        df.groupby(["month_id", COL_RECEIVER_ID], observed=True)
        .agg(
            n_tx_recv=(COL_AMOUNT, "count"),
            sum_recv=(COL_AMOUNT, "sum"),
            avg_recv=(COL_AMOUNT, "mean"),
            risk_avg_re=("risk_score", "mean"),
            risk_max_recv=("risk_score", "max"),
        )
        .rename_axis(index={COL_RECEIVER_ID: "persona"})
        .reset_index()
    )
    people = pd.merge(em, re, on=["month_id", "persona"], how="outer").fillna(
        {
            "n_tx_emit": 0,
            "sum_emit": 0.0,
            "avg_emit": 0.0,
            "risk_avg_em": 0.0,
            "risk_max_emit": 0.0,
            "n_tx_recv": 0,
            "sum_recv": 0.0,
            "avg_recv": 0.0,
            "risk_avg_re": 0.0,
            "risk_max_recv": 0.0,
        }
    )
    people["risk_avg_person"] = (people["risk_avg_em"] + people["risk_avg_re"]) / 2.0
    topc = (
        df[df["nlp_concepto_sospechoso"] != ""]
        .groupby(["month_id", COL_SENDER_ID, "nlp_concepto_sospechoso"], observed=True)[COL_AMOUNT]
        .count()
        .reset_index(name="cnt")
    )
    topc = (
        topc.sort_values(["month_id", COL_SENDER_ID, "cnt"], ascending=[True, True, False])
        .groupby(["month_id", COL_SENDER_ID])
        .head(3)
    )
    tops = (
        topc.groupby(["month_id", COL_SENDER_ID])["nlp_concepto_sospechoso"].apply(list).rename("top_conceptos")
    )
    people = people.merge(
        tops,
        left_on=["month_id", "persona"],
        right_on=["month_id", COL_SENDER_ID],
        how="left",
    ).drop(columns=[COL_SENDER_ID], errors="ignore")
    people["top_conceptos"] = people["top_conceptos"].apply(lambda x: x if isinstance(x, list) else [])
    people["interp_person"] = people.apply(person_interpretation, axis=1)
    return people.sort_values(["risk_avg_person", "sum_emit", "sum_recv"], ascending=[False, False, False])
