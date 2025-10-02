import pandas as pd

from ..interpret.person import person_interpretation
from ..schemas import (
    COL_AMOUNT,
    COL_RECEIVER_ID,
    COL_RECEIVER_TENURE_YEARS,
    COL_SENDER_ID,
    COL_SENDER_TENURE_YEARS,
)


CASE13_NEW_EMPLOYEE_YEARS = 0.5
CASE13_MIN_HIGH_TX = 2
CASE13_MIN_UNIQUE_SENDERS = 3
CASE14_NEW_EMPLOYEE_YEARS = 0.5
CASE14_OLD_EMPLOYEE_YEARS = 5.0
CASE14_MIN_TX = 5
CASE14_MIN_UNIQUE_SENDERS = 3


def _ensure_string(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.strip()


def _safe_zscore(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index, dtype="float64")
    return (values - mean) / std


def _prepare_case_base(df: pd.DataFrame) -> pd.DataFrame:
    needed = [
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        COL_AMOUNT,
        COL_SENDER_TENURE_YEARS,
        COL_RECEIVER_TENURE_YEARS,
    ]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        return pd.DataFrame(columns=needed)
    base = df[needed].copy()
    base[COL_AMOUNT] = pd.to_numeric(base[COL_AMOUNT], errors="coerce")
    base[COL_SENDER_TENURE_YEARS] = pd.to_numeric(
        base[COL_SENDER_TENURE_YEARS], errors="coerce"
    )
    base[COL_RECEIVER_TENURE_YEARS] = pd.to_numeric(
        base[COL_RECEIVER_TENURE_YEARS], errors="coerce"
    )
    base = base.dropna(subset=[COL_AMOUNT])
    return base


def _compute_case13_new_receivers(df: pd.DataFrame) -> pd.DataFrame:
    base = _prepare_case_base(df)
    if base.empty:
        return pd.DataFrame(
            columns=[
                "persona",
                "caso13_persona_tx_recibidas",
                "caso13_persona_monto_total",
                "caso13_persona_emisores_unicos",
                "caso13_persona_tx_altos",
                "caso13_persona_monto_promedio",
                "caso13_persona_flag_nuevo_receptor_altos_montos",
            ]
        )

    amounts = base[COL_AMOUNT].dropna()
    if amounts.empty:
        high_amount_thr = float("inf")
    else:
        high_amount_thr = float(amounts.quantile(0.9))

    base["is_high_amount"] = base[COL_AMOUNT] >= high_amount_thr
    new_receivers = base[
        base[COL_RECEIVER_TENURE_YEARS].fillna(float("inf")) <= CASE13_NEW_EMPLOYEE_YEARS
    ]
    if new_receivers.empty:
        return pd.DataFrame(
            columns=[
                "persona",
                "caso13_persona_tx_recibidas",
                "caso13_persona_monto_total",
                "caso13_persona_emisores_unicos",
                "caso13_persona_tx_altos",
                "caso13_persona_monto_promedio",
                "caso13_persona_flag_nuevo_receptor_altos_montos",
            ]
        )

    agg = (
        new_receivers.groupby(COL_RECEIVER_ID, observed=True)
        .agg(
            caso13_persona_tx_recibidas=(COL_AMOUNT, "count"),
            caso13_persona_monto_total=(COL_AMOUNT, "sum"),
            caso13_persona_emisores_unicos=(COL_SENDER_ID, "nunique"),
            caso13_persona_tx_altos=("is_high_amount", "sum"),
            caso13_persona_monto_promedio=(COL_AMOUNT, "mean"),
        )
        .reset_index()
        .rename(columns={COL_RECEIVER_ID: "persona"})
    )

    agg["caso13_persona_flag_nuevo_receptor_altos_montos"] = (
        (agg["caso13_persona_emisores_unicos"] >= CASE13_MIN_UNIQUE_SENDERS)
        & (agg["caso13_persona_tx_altos"] >= CASE13_MIN_HIGH_TX)
    ).astype(int)
    return agg


def _compute_case14_old_receivers(df: pd.DataFrame) -> pd.DataFrame:
    base = _prepare_case_base(df)
    if base.empty:
        return pd.DataFrame(
            columns=[
                "persona",
                "caso14_persona_tx_de_emisores_nuevos",
                "caso14_persona_monto_de_emisores_nuevos",
                "caso14_persona_emisores_nuevos_unicos",
                "caso14_persona_monto_promedio_de_emisores_nuevos",
                "caso14_persona_flag_antiguo_recibe_de_nuevos",
            ]
        )

    mask = (
        base[COL_SENDER_TENURE_YEARS].fillna(float("inf")) <= CASE14_NEW_EMPLOYEE_YEARS
    ) & (
        base[COL_RECEIVER_TENURE_YEARS].fillna(-float("inf"))
        >= CASE14_OLD_EMPLOYEE_YEARS
    )
    filtered = base.loc[mask].copy()
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "persona",
                "caso14_persona_tx_de_emisores_nuevos",
                "caso14_persona_monto_de_emisores_nuevos",
                "caso14_persona_emisores_nuevos_unicos",
                "caso14_persona_monto_promedio_de_emisores_nuevos",
                "caso14_persona_flag_antiguo_recibe_de_nuevos",
            ]
        )

    agg = (
        filtered.groupby(COL_RECEIVER_ID, observed=True)
        .agg(
            caso14_persona_tx_de_emisores_nuevos=(COL_AMOUNT, "count"),
            caso14_persona_monto_de_emisores_nuevos=(COL_AMOUNT, "sum"),
            caso14_persona_emisores_nuevos_unicos=(COL_SENDER_ID, "nunique"),
            caso14_persona_monto_promedio_de_emisores_nuevos=(COL_AMOUNT, "mean"),
        )
        .reset_index()
        .rename(columns={COL_RECEIVER_ID: "persona"})
    )

    agg["caso14_persona_flag_antiguo_recibe_de_nuevos"] = (
        (agg["caso14_persona_emisores_nuevos_unicos"] >= CASE14_MIN_UNIQUE_SENDERS)
        & (agg["caso14_persona_tx_de_emisores_nuevos"] >= CASE14_MIN_TX)
    ).astype(int)
    return agg


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
                "nlp_persona_score_emit_promedio",
                "nlp_persona_sentimiento_emit_promedio",
                "nlp_persona_score_recv_promedio",
                "nlp_persona_sentimiento_recv_promedio",
                "nlp_persona_score_prob_coi",
                "nlp_persona_sentimiento_promedio",
                "desbalance_persona_monto_neto",
                "desbalance_persona_ratio_emision_vs_recepcion",
                "desbalance_persona_zscore_emision_total",
                "desbalance_persona_zscore_recepcion_total",
                "desbalance_persona_zscore_neto_total",
                "desbalance_persona_meses_totales",
                "desbalance_persona_meses_envia_extremo",
                "desbalance_persona_meses_recibe_extremo",
                "desbalance_persona_tasa_meses_envia_extremo",
                "desbalance_persona_tasa_meses_recibe_extremo",
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
                "caso13_persona_tx_recibidas",
                "caso13_persona_monto_total",
                "caso13_persona_emisores_unicos",
                "caso13_persona_tx_altos",
                "caso13_persona_monto_promedio",
                "caso13_persona_flag_nuevo_receptor_altos_montos",
                "caso14_persona_tx_de_emisores_nuevos",
                "caso14_persona_monto_de_emisores_nuevos",
                "caso14_persona_emisores_nuevos_unicos",
                "caso14_persona_monto_promedio_de_emisores_nuevos",
                "caso14_persona_flag_antiguo_recibe_de_nuevos",
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

    score_series = df.get("feat_nlp_coi_score")
    if score_series is None:
        score_series = pd.Series(0.0, index=df.index, dtype="float64")
    else:
        score_series = pd.to_numeric(score_series, errors="coerce").fillna(0.0).astype(float)
    senti_series = df.get("feat_nlp_sentimiento")
    if senti_series is None:
        senti_series = pd.Series(0.0, index=df.index, dtype="float64")
    else:
        senti_series = pd.to_numeric(senti_series, errors="coerce").fillna(0.0).astype(float)

    if COL_SENDER_ID in df.columns:
        emit_payload = pd.DataFrame(
            {
                "persona": df[COL_SENDER_ID].astype("string"),
                "_score": score_series.values,
                "_sent": senti_series.values,
            }
        )
        emit_stats = (
            emit_payload.groupby("persona", observed=True)
            .agg(
                nlp_persona_score_emit_promedio=("_score", "mean"),
                nlp_persona_score_emit_max=("_score", "max"),
                nlp_persona_sentimiento_emit_promedio=("_sent", "mean"),
            )
            .reset_index()
        )
    else:
        emit_stats = pd.DataFrame(
            columns=[
                "persona",
                "nlp_persona_score_emit_promedio",
                "nlp_persona_score_emit_max",
                "nlp_persona_sentimiento_emit_promedio",
            ]
        )

    if COL_RECEIVER_ID in df.columns:
        recv_payload = pd.DataFrame(
            {
                "persona": df[COL_RECEIVER_ID].astype("string"),
                "_score": score_series.values,
                "_sent": senti_series.values,
            }
        )
        recv_stats = (
            recv_payload.groupby("persona", observed=True)
            .agg(
                nlp_persona_score_recv_promedio=("_score", "mean"),
                nlp_persona_score_recv_max=("_score", "max"),
                nlp_persona_sentimiento_recv_promedio=("_sent", "mean"),
            )
            .reset_index()
        )
    else:
        recv_stats = pd.DataFrame(
            columns=[
                "persona",
                "nlp_persona_score_recv_promedio",
                "nlp_persona_score_recv_max",
                "nlp_persona_sentimiento_recv_promedio",
            ]
        )

    people = people.merge(emit_stats, on="persona", how="left")
    people = people.merge(recv_stats, on="persona", how="left")

    for col in [
        "nlp_persona_score_emit_promedio",
        "nlp_persona_score_emit_max",
        "nlp_persona_sentimiento_emit_promedio",
        "nlp_persona_score_recv_promedio",
        "nlp_persona_score_recv_max",
        "nlp_persona_sentimiento_recv_promedio",
    ]:
        if col not in people:
            people[col] = 0.0
        else:
            people[col] = people[col].fillna(0.0).astype(float)

    max_cols = [
        col
        for col in [
            "nlp_persona_score_emit_promedio",
            "nlp_persona_score_recv_promedio",
            "nlp_persona_score_emit_max",
            "nlp_persona_score_recv_max",
        ]
        if col in people
    ]
    if max_cols:
        people["nlp_persona_score_prob_coi"] = people[max_cols].max(axis=1)
    else:
        people["nlp_persona_score_prob_coi"] = 0.0

    emit_sent = people.get("nlp_persona_sentimiento_emit_promedio", pd.Series(0.0, index=people.index))
    recv_sent = people.get("nlp_persona_sentimiento_recv_promedio", pd.Series(0.0, index=people.index))
    emit_mask = (emit_sent.notna()).astype(int)
    recv_mask = (recv_sent.notna()).astype(int)
    emit_sent = emit_sent.fillna(0.0)
    recv_sent = recv_sent.fillna(0.0)
    denom = (emit_mask + recv_mask).replace(0, 1)
    people["nlp_persona_sentimiento_promedio"] = (emit_sent + recv_sent) / denom

    drop_cols = ["nlp_persona_score_emit_max", "nlp_persona_score_recv_max"]
    existing_drop = [c for c in drop_cols if c in people]
    if existing_drop:
        people = people.drop(columns=existing_drop)


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
    people["desbalance_persona_zscore_emision_total"] = _safe_zscore(people["sum_emit"])
    people["desbalance_persona_zscore_recepcion_total"] = _safe_zscore(people["sum_recv"])
    people["desbalance_persona_zscore_neto_total"] = _safe_zscore(
        people["desbalance_persona_monto_neto"]
    )

    monthly_cols = [
        "desbalance_persona_meses_totales",
        "desbalance_persona_meses_envia_extremo",
        "desbalance_persona_meses_recibe_extremo",
        "desbalance_persona_tasa_meses_envia_extremo",
        "desbalance_persona_tasa_meses_recibe_extremo",
    ]
    if "month_id" in df.columns:
        em_month = (
            df.groupby(["month_id", COL_SENDER_ID], observed=True)
            .agg(
                sum_emit_mes=(COL_AMOUNT, "sum"),
                n_tx_emit_mes=(COL_AMOUNT, "count"),
            )
            .reset_index()
            .rename(columns={COL_SENDER_ID: "persona"})
        )
        re_month = (
            df.groupby(["month_id", COL_RECEIVER_ID], observed=True)
            .agg(
                sum_recv_mes=(COL_AMOUNT, "sum"),
                n_tx_recv_mes=(COL_AMOUNT, "count"),
            )
            .reset_index()
            .rename(columns={COL_RECEIVER_ID: "persona"})
        )
        monthly = em_month.merge(re_month, on=["month_id", "persona"], how="outer").fillna(
            {
                "sum_emit_mes": 0.0,
                "n_tx_emit_mes": 0,
                "sum_recv_mes": 0.0,
                "n_tx_recv_mes": 0,
            }
        )
        monthly = monthly[
            (monthly["n_tx_emit_mes"] > 0) | (monthly["n_tx_recv_mes"] > 0)
        ].copy()
        if not monthly.empty:
            monthly["desbalance_mes_neto"] = (
                monthly["sum_emit_mes"] - monthly["sum_recv_mes"]
            )
            monthly["z_emit_mes"] = monthly.groupby("month_id", observed=True)[
                "sum_emit_mes"
            ].transform(_safe_zscore)
            monthly["z_recv_mes"] = monthly.groupby("month_id", observed=True)[
                "sum_recv_mes"
            ].transform(_safe_zscore)
            monthly["z_neto_mes"] = monthly.groupby("month_id", observed=True)[
                "desbalance_mes_neto"
            ].transform(_safe_zscore)
            monthly["flag_envia_extremo"] = (
                (monthly["desbalance_mes_neto"] > 0)
                & (monthly["z_neto_mes"].abs() >= 2.0)
            )
            monthly["flag_recibe_extremo"] = (
                (monthly["desbalance_mes_neto"] < 0)
                & (monthly["z_neto_mes"].abs() >= 2.0)
            )
            monthly_stats = (
                monthly.groupby("persona", observed=True)
                .agg(
                    desbalance_persona_meses_totales=("month_id", "nunique"),
                    desbalance_persona_meses_envia_extremo=("flag_envia_extremo", "sum"),
                    desbalance_persona_meses_recibe_extremo=("flag_recibe_extremo", "sum"),
                )
                .reset_index()
            )
            monthly_stats["desbalance_persona_tasa_meses_envia_extremo"] = (
                monthly_stats["desbalance_persona_meses_envia_extremo"]
                / monthly_stats["desbalance_persona_meses_totales"].replace({0: pd.NA})
            ).fillna(0.0)
            monthly_stats["desbalance_persona_tasa_meses_recibe_extremo"] = (
                monthly_stats["desbalance_persona_meses_recibe_extremo"]
                / monthly_stats["desbalance_persona_meses_totales"].replace({0: pd.NA})
            ).fillna(0.0)
            people = people.merge(monthly_stats, on="persona", how="left")
    for col in monthly_cols:
        if col not in people:
            people[col] = 0
    people["desbalance_persona_meses_totales"] = (
        people["desbalance_persona_meses_totales"].fillna(0).astype("Int64")
    )
    people["desbalance_persona_meses_envia_extremo"] = (
        people["desbalance_persona_meses_envia_extremo"].fillna(0).astype("Int64")
    )
    people["desbalance_persona_meses_recibe_extremo"] = (
        people["desbalance_persona_meses_recibe_extremo"].fillna(0).astype("Int64")
    )
    people["desbalance_persona_tasa_meses_envia_extremo"] = (
        people["desbalance_persona_tasa_meses_envia_extremo"].fillna(0.0).astype(float)
    )
    people["desbalance_persona_tasa_meses_recibe_extremo"] = (
        people["desbalance_persona_tasa_meses_recibe_extremo"].fillna(0.0).astype(float)
    )

    case13 = _compute_case13_new_receivers(df)
    if not case13.empty:
        people = people.merge(case13, on="persona", how="left")
    case14 = _compute_case14_old_receivers(df)
    if not case14.empty:
        people = people.merge(case14, on="persona", how="left")

    for col, fill_value in [
        ("caso13_persona_tx_recibidas", 0),
        ("caso13_persona_monto_total", 0.0),
        ("caso13_persona_emisores_unicos", 0),
        ("caso13_persona_tx_altos", 0),
        ("caso13_persona_monto_promedio", 0.0),
        ("caso13_persona_flag_nuevo_receptor_altos_montos", 0),
        ("caso14_persona_tx_de_emisores_nuevos", 0),
        ("caso14_persona_monto_de_emisores_nuevos", 0.0),
        ("caso14_persona_emisores_nuevos_unicos", 0),
        ("caso14_persona_monto_promedio_de_emisores_nuevos", 0.0),
        ("caso14_persona_flag_antiguo_recibe_de_nuevos", 0),
    ]:
        if col not in people:
            people[col] = fill_value
        else:
            people[col] = people[col].fillna(fill_value)

    if "nlp_persona_top_conceptos" in people:
        people["top_conceptos"] = people["nlp_persona_top_conceptos"]
    else:
        people["top_conceptos"] = [[] for _ in range(len(people))]
    people["top_conceptos"] = people["top_conceptos"].apply(lambda x: x if isinstance(x, list) else [])
    people["interp_person"] = people.apply(person_interpretation, axis=1)
    return people.sort_values(
        ["risk_avg_person", "sum_emit", "sum_recv"], ascending=[False, False, False]
    )
