import pandas as pd

from ..interpret.person import person_interpretation
from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_SENDER_ID


def _ensure_string(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.strip()


def build_person_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "persona",
                "n_tx_emit",
                "sum_emit",
                "avg_emit",
                "risk_avg_em",
                "risk_max_emit",
                "n_tx_recv",
                "sum_recv",
                "avg_recv",
                "risk_avg_re",
                "risk_max_recv",
                "risk_avg_person",
                "top_conceptos",
                "interp_person",
                "nlp_persona_total_transacciones_sospechosas",
                "nlp_persona_conceptos_sospechosos_unicos",
                "nlp_persona_top_conceptos",
                "desbalance_persona_monto_neto",
                "desbalance_persona_ratio_emision_vs_recepcion",
                "yo_yo_persona_tasa_flag_emisor",
                "smurf_persona_tasa_flag_emisor",
                "frecuencia_persona_tasa_flag_emisor",
                "recurrente_persona_tasa_flag_emisor",
                "prestamo_persona_tasa_repay_insuficiente",
                "monto_persona_tasa_flag_redondo",
                "umbral_persona_tasa_flag_cercania",
                "red_persona_tasa_en_ciclos",
                "red_persona_tasa_en_triangulos",
                "quid_pro_quo_persona_tasa_flag",
                "referencia_persona_tasa_reutilizada",
                "cambio_brusco_persona_tasa_flag",
                "nuevo_enlace_persona_tasa_flag",
            ]
        )

    df = df.copy()
    for col in [
        "sig_quid_pro_quo",
        "sig_reference_reuse",
        "sig_pair_change_point",
        "sig_pair_new_edge",
    ]:
        if col not in df:
            df[col] = 0.0

    em = (
        df.groupby(COL_SENDER_ID, observed=True)
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
        df.groupby(COL_RECEIVER_ID, observed=True)
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
    people = pd.merge(em, re, on="persona", how="outer").fillna(
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

    concept_series = _ensure_string(df.get("nlp_concepto_sospechoso", ""))
    df_concepts = df.loc[concept_series != ""].copy()
    df_concepts["nlp_concepto_sospechoso"] = concept_series.loc[df_concepts.index]

    if not df_concepts.empty:
        suspicious_counts = (
            df_concepts.groupby(COL_SENDER_ID, observed=True)["nlp_concepto_sospechoso"]
            .agg(
                nlp_persona_total_transacciones_sospechosas="count",
                nlp_persona_conceptos_sospechosos_unicos="nunique",
            )
            .reset_index()
            .rename(columns={COL_SENDER_ID: "persona"})
        )
        topc = (
            df_concepts.groupby([COL_SENDER_ID, "nlp_concepto_sospechoso"], observed=True)[
                COL_AMOUNT
            ]
            .count()
            .reset_index(name="cnt")
        )
        topc = (
            topc.sort_values([COL_SENDER_ID, "cnt"], ascending=[True, False])
            .groupby(COL_SENDER_ID)
            .head(3)
        )
        tops = (
            topc.groupby(COL_SENDER_ID)["nlp_concepto_sospechoso"]
            .apply(list)
            .rename("nlp_persona_top_conceptos")
            .reset_index()
            .rename(columns={COL_SENDER_ID: "persona"})
        )
        people = people.merge(suspicious_counts, on="persona", how="left")
        people = people.merge(tops, on="persona", how="left")
    else:
        people["nlp_persona_total_transacciones_sospechosas"] = 0
        people["nlp_persona_conceptos_sospechosos_unicos"] = 0
        people["nlp_persona_top_conceptos"] = [[] for _ in range(len(people))]

    for col in [
        "nlp_persona_total_transacciones_sospechosas",
        "nlp_persona_conceptos_sospechosos_unicos",
    ]:
        if col not in people:
            people[col] = 0
        else:
            people[col] = people[col].fillna(0)
    if "nlp_persona_top_conceptos" not in people:
        people["nlp_persona_top_conceptos"] = [[] for _ in range(len(people))]
    else:
        people["nlp_persona_top_conceptos"] = people["nlp_persona_top_conceptos"].apply(
            lambda x: x if isinstance(x, list) else ([] if pd.isna(x) else [x])
        )

    flag_stats = (
        df.groupby(COL_SENDER_ID, observed=True)
        .agg(
            yo_yo_persona_tasa_flag_emisor=("sig_yoyo", "mean"),
            smurf_persona_tasa_flag_emisor=("sig_smurf", "mean"),
            frecuencia_persona_tasa_flag_emisor=("sig_freq", "mean"),
            recurrente_persona_tasa_flag_emisor=("sig_recurrent", "mean"),
            prestamo_persona_tasa_repay_insuficiente=("sig_loan_bad_repay", "mean"),
            monto_persona_tasa_flag_redondo=("sig_roundsum", "mean"),
            umbral_persona_tasa_flag_cercania=("sig_near_thr", "mean"),
            red_persona_tasa_en_ciclos=("p1_in_cycle", "mean"),
            red_persona_tasa_en_triangulos=("p1_in_triangle", "mean"),
            quid_pro_quo_persona_tasa_flag=("sig_quid_pro_quo", "mean"),
            referencia_persona_tasa_reutilizada=("sig_reference_reuse", "mean"),
            cambio_brusco_persona_tasa_flag=("sig_pair_change_point", "mean"),
            nuevo_enlace_persona_tasa_flag=("sig_pair_new_edge", "mean"),
        )
        .reset_index()
        .rename(columns={COL_SENDER_ID: "persona"})
    )
    people = people.merge(flag_stats, on="persona", how="left")

    fill_cols = [
        "yo_yo_persona_tasa_flag_emisor",
        "smurf_persona_tasa_flag_emisor",
        "frecuencia_persona_tasa_flag_emisor",
        "recurrente_persona_tasa_flag_emisor",
        "prestamo_persona_tasa_repay_insuficiente",
        "monto_persona_tasa_flag_redondo",
        "umbral_persona_tasa_flag_cercania",
        "red_persona_tasa_en_ciclos",
        "red_persona_tasa_en_triangulos",
        "quid_pro_quo_persona_tasa_flag",
        "referencia_persona_tasa_reutilizada",
        "cambio_brusco_persona_tasa_flag",
        "nuevo_enlace_persona_tasa_flag",
    ]
    for col in fill_cols:
        if col not in people:
            people[col] = 0.0
        else:
            people[col] = people[col].fillna(0.0)

    people["desbalance_persona_monto_neto"] = people["sum_emit"] - people["sum_recv"]
    total_mov = people["sum_emit"] + people["sum_recv"]
    people["desbalance_persona_ratio_emision_vs_recepcion"] = (
        people["sum_emit"].where(total_mov == 0, people["sum_emit"] / total_mov)
    )

    if "nlp_persona_top_conceptos" in people:
        people["top_conceptos"] = people["nlp_persona_top_conceptos"]
    else:
        people["top_conceptos"] = [[] for _ in range(len(people))]
    people["top_conceptos"] = people["top_conceptos"].apply(lambda x: x if isinstance(x, list) else [])
    people["interp_person"] = people.apply(person_interpretation, axis=1)
    return people.sort_values(
        ["risk_avg_person", "sum_emit", "sum_recv"], ascending=[False, False, False]
    )
