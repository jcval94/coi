"""Visualizaciones con Seaborn para las preguntas de `experiment_questions`."""
from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from coi_fraud.schemas import COL_RECEIVER_ID, COL_SENDER_ID
from experiment_questions import (
    DEFAULT_TIMEFRAME,
    question1_manager_nlp,
    question2_manager_concepts,
    question3_quid_pairs,
    question4_quid_negative_value_vs_load,
    question5_reference_reuse,
    question6_centralizers,
    question7_net_imbalance,
    question8_case13_new_employees,
    question9_case14_veterans_from_newcomers,
    question10_yoyo_streaks,
    question11_near_threshold_structuring,
    question12_smurfing_chronic,
    question13_bad_loans_with_frequency,
    question14_recurrent_payroll,
    question15_coordinated_cluster_signals,
    question16_multisignal_transactions,
    question17_nlp_person_profiles,
)


sns.set_theme(style="whitegrid")


def _ensure_axis(ax: Optional[Axes] = None, figsize: tuple[int, int] = (8, 5)) -> Axes:
    if ax is not None:
        return ax
    _, created_ax = plt.subplots(figsize=figsize)
    return created_ax


def _empty_chart(ax: Axes, title: str, message: str) -> Axes:
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return ax


def plot_q1_manager_nlp(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica los pares manager-subordinado con conceptos NLP sospechosos."""
    data = question1_manager_nlp(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q1 – Conceptos sospechosos",
            "Sin coincidencias manager-subordinado para el periodo seleccionado.",
        )

    work = data.copy()
    work["pair"] = (
        work["manager_user_id"].fillna("sin_manager").astype(str)
        + "→"
        + work["subordinado_user_id"].fillna("sin_subordinado").astype(str)
    )
    work = work.sort_values(["tx_count", "monto_total"], ascending=[False, False]).head(top_n)

    sns.barplot(data=work, x="tx_count", y="pair", hue="nlp_concepto_sospechoso", ax=axis)
    axis.set_title(f"Q1 – Pares con conceptos sospechosos ({timeframe})")
    axis.set_xlabel("Número de transacciones")
    axis.set_ylabel("Manager → Subordinado")
    axis.legend(title="Concepto")
    return axis


def plot_q2_manager_concepts(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 8,
) -> Axes:
    """Grafica conceptos NLP por severidad (P95)."""
    data = question2_manager_concepts(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q2 – Severidad NLP",
            "Sin conceptos con severidad calculada para el periodo.",
        )

    work = data.copy()
    work = work.sort_values(["risk_p95", "tx_count"], ascending=[False, False]).head(top_n)
    plot_kwargs = {
        "data": work,
        "x": "risk_p95",
        "y": "nlp_concepto_sospechoso",
        "ax": axis,
    }
    if "month_id" in work:
        plot_kwargs["hue"] = "month_id"
    sns.barplot(**plot_kwargs)
    if axis.get_legend() is not None:
        axis.legend(title="Mes")
    axis.set_title(f"Q2 – Severidad por concepto ({timeframe})")
    axis.set_xlabel("P95 de risk_score")
    axis.set_ylabel("Concepto NLP")
    return axis


def plot_q3_quid_pairs(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica pares destacados por puntaje quid-pro-quo."""
    data = question3_quid_pairs(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q3 – Pares Quid Pro Quo",
            "Sin pares destacados para el periodo.",
        )

    work = data.copy()
    if "nivel_respuesta" in work.columns:
        work = work.loc[work["nivel_respuesta"] == "par"].copy()
    if work.empty:
        return _empty_chart(
            axis,
            "Q3 – Pares Quid Pro Quo",
            "Solo se encontraron detalles de transacción, no pares agregados.",
        )

    work["label"] = (
        work.get("quid_pair_label", work.get("quid_pair_clave", "sin_par"))
        .fillna("sin_par")
        .astype(str)
    )
    work = work.sort_values(["quid_score_max", "quid_tx_count"], ascending=[False, False]).head(top_n)
    sns.barplot(data=work, x="quid_score_max", y="label", hue="quid_tx_count", ax=axis)
    axis.set_title(f"Q3 – Pares con mayor puntaje ({timeframe})")
    axis.set_xlabel("Puntaje máximo quid-pro-quo")
    axis.set_ylabel("Par emisor → receptor")
    axis.legend(title="Transacciones")
    return axis


def plot_q4_negative_value_vs_load(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica transacciones con desfase negativo autorización vs. carga."""
    data = question4_quid_negative_value_vs_load(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q4 – Autorización vs. carga",
            "Sin transacciones con desfase negativo registradas.",
        )

    work = data.copy()
    work = work.sort_values("feat_quid_value_vs_load_days", ascending=True).head(top_n)
    work["pair"] = (
        work.get(COL_SENDER_ID, pd.Series(dtype="object")).fillna("sin_emisor").astype(str)
        + "→"
        + work.get(COL_RECEIVER_ID, pd.Series(dtype="object")).fillna("sin_receptor").astype(str)
    )
    plot_kwargs = {
        "data": work,
        "x": "feat_quid_value_vs_load_days",
        "y": "pair",
        "ax": axis,
    }
    if "feat_quid_score" in work:
        plot_kwargs["hue"] = "feat_quid_score"
    sns.barplot(**plot_kwargs)
    if axis.get_legend() is not None:
        axis.legend(title="Puntaje")
    axis.set_title(f"Q4 – Desfase autorización/carga ({timeframe})")
    axis.set_xlabel("Días (negativos = autorización previa)")
    axis.set_ylabel("Par emisor → receptor")
    return axis


def plot_q5_reference_reuse(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica referencias de pago recurrentes."""
    data = question5_reference_reuse(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q5 – Reutilización de referencias",
            "Sin referencias recurrentes detectadas.",
        )

    work = data.copy()
    if "nivel_respuesta" in work.columns:
        work = work.loc[work["nivel_respuesta"] == "referencia"].copy()
    if work.empty:
        return _empty_chart(
            axis,
            "Q5 – Reutilización de referencias",
            "Solo hay detalle de transacciones sin resumen por referencia.",
        )

    work = work.sort_values(["tx_count", "n_pairs"], ascending=[False, False]).head(top_n)
    if "reference_norm" in work:
        work["reference_norm"] = work["reference_norm"].fillna("sin_referencia").astype(str)
    sns.barplot(data=work, x="tx_count", y="reference_norm", hue="n_pairs", ax=axis)
    axis.set_title(f"Q5 – Referencias más reutilizadas ({timeframe})")
    axis.set_xlabel("Número de transacciones")
    axis.set_ylabel("Referencia normalizada")
    axis.legend(title="Pares distintos")
    return axis


def plot_q6_centralizers(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica receptores con mayor centralidad."""
    data = question6_centralizers(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q6 – Receptores centralizadores",
            "Sin receptores centralizadores para el periodo.",
        )

    work = data.copy()
    work = work.sort_values(["centralidad", "inflow"], ascending=[False, False]).head(top_n)
    work[COL_RECEIVER_ID] = work[COL_RECEIVER_ID].fillna("sin_receptor").astype(str)
    plot_kwargs = {
        "data": work,
        "x": "centralidad",
        "y": COL_RECEIVER_ID,
        "ax": axis,
    }
    if "month_id" in work:
        plot_kwargs["hue"] = "month_id"
    sns.barplot(**plot_kwargs)
    if axis.get_legend() is not None:
        axis.legend(title="Mes")
    axis.set_title(f"Q6 – Receptores centralizadores ({timeframe})")
    axis.set_xlabel("Centralidad (inflow × emisores únicos)")
    axis.set_ylabel("Receptor")
    return axis


def plot_q7_net_imbalance(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica personas con mayor desbalance neto."""
    data = question7_net_imbalance(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q7 – Desbalance neto",
            "Sin personas con desbalance calculado.",
        )

    work = data.copy()
    work["persona"] = work["persona"].fillna("sin_persona").astype(str)
    work["abs_neto"] = work["desbalance_persona_monto_neto"].abs()
    work = work.sort_values("abs_neto", ascending=False).head(top_n)
    sns.barplot(data=work, x="desbalance_persona_monto_neto", y="persona", ax=axis)
    axis.set_title(f"Q7 – Personas con desbalance neto ({timeframe})")
    axis.set_xlabel("Monto neto (positivo = recibe más)")
    axis.set_ylabel("Persona")
    return axis


def plot_q8_case13_new_employees(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica receptores nuevos con montos altos recibidos."""
    data = question8_case13_new_employees(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q8 – Receptores nuevos",
            "Sin receptores recientes con montos altos.",
        )

    work = data.copy()
    work["persona"] = work["persona"].fillna("sin_persona").astype(str)
    work = work.sort_values(
        ["caso13_persona_tx_altos", "caso13_persona_monto_total"],
        ascending=[False, False],
    ).head(top_n)
    sns.barplot(
        data=work,
        x="caso13_persona_monto_total",
        y="persona",
        hue="caso13_persona_tx_altos",
        ax=axis,
    )
    axis.set_title(f"Q8 – Receptores nuevos con montos altos ({timeframe})")
    axis.set_xlabel("Monto total recibido")
    axis.set_ylabel("Receptor")
    axis.legend(title="Tx monto alto")
    return axis


def plot_q9_case14_veterans_from_newcomers(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica veteranos que reciben de emisores recientes."""
    data = question9_case14_veterans_from_newcomers(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q9 – Veteranos desde nuevos",
            "Sin veteranos recibiendo de emisores nuevos.",
        )

    work = data.copy()
    work["persona"] = work["persona"].fillna("sin_persona").astype(str)
    work = work.sort_values(
        ["caso14_persona_tx_de_emisores_nuevos", "caso14_persona_monto_de_emisores_nuevos"],
        ascending=[False, False],
    ).head(top_n)
    sns.barplot(
        data=work,
        x="caso14_persona_monto_de_emisores_nuevos",
        y="persona",
        hue="caso14_persona_emisores_nuevos_unicos",
        ax=axis,
    )
    axis.set_title(f"Q9 – Veteranos receptores de nuevos ({timeframe})")
    axis.set_xlabel("Monto recibido desde emisores nuevos")
    axis.set_ylabel("Receptor veterano")
    axis.legend(title="Emisores únicos")
    return axis


def plot_q10_yoyo_streaks(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica pares con mayores rachas Yo-Yo y riesgo."""
    data = question10_yoyo_streaks(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q10 – Rachas Yo-Yo",
            "Sin rachas Yo-Yo identificadas.",
        )

    work = data.copy()
    work["par_bidir"] = work["par_bidir"].fillna("sin_par").astype(str)
    work = work.sort_values(
        ["racha_max_yo_yo", "riesgo_max_par", "tx_yo_yo_totales"],
        ascending=[False, False, False],
    ).head(top_n)
    sns.scatterplot(
        data=work,
        x="racha_max_yo_yo",
        y="riesgo_max_par",
        size="tx_yo_yo_totales",
        hue="meses_con_yo_yo",
        ax=axis,
    )
    for _, row in work.iterrows():
        axis.text(
            row["racha_max_yo_yo"],
            row["riesgo_max_par"],
            row["par_bidir"],
            fontsize=8,
            ha="left",
        )
    axis.set_title(f"Q10 – Rachas Yo-Yo ({timeframe})")
    axis.set_xlabel("Racha máxima Yo-Yo")
    axis.set_ylabel("Riesgo máximo del par")
    axis.legend(title="Meses Yo-Yo", loc="best")
    return axis


def plot_q11_near_threshold_structuring(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica pares con operaciones cerca de umbrales regulados."""
    data = question11_near_threshold_structuring(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q11 – Cercanos a umbral",
            "Sin montos cercanos a umbrales.",
        )

    work = data.copy()
    work["pair"] = work["pair"].fillna("sin_par").astype(str)
    work = work.sort_values(
        ["meses_con_near", "monto_total_near"],
        ascending=[False, False],
    ).head(top_n)
    sns.barplot(
        data=work,
        x="meses_con_near",
        y="pair",
        hue="riesgo_max",
        ax=axis,
    )
    axis.set_title(f"Q11 – Montos pegados al umbral ({timeframe})")
    axis.set_xlabel("Meses con montos cercanos")
    axis.set_ylabel("Par emisor → receptor")
    axis.legend(title="Riesgo máximo")
    return axis


def plot_q12_smurfing_chronic(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica pares con patrones crónicos de smurfing."""
    data = question12_smurfing_chronic(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q12 – Smurfing crónico",
            "Sin pares con smurfing prolongado.",
        )

    work = data.copy()
    work["pair"] = work["pair"].fillna("sin_par").astype(str)
    work = work.sort_values(
        ["meses_con_smurf", "monto_smurf_total"],
        ascending=[False, False],
    ).head(top_n)
    sns.scatterplot(
        data=work,
        x="meses_con_smurf",
        y="monto_smurf_total",
        size="tx_smurf_totales",
        hue="riesgo_max",
        ax=axis,
    )
    for _, row in work.iterrows():
        axis.text(
            row["meses_con_smurf"],
            row["monto_smurf_total"],
            row["pair"],
            fontsize=8,
            ha="left",
        )
    axis.set_title(f"Q12 – Smurfing crónico ({timeframe})")
    axis.set_xlabel("Meses con smurfing")
    axis.set_ylabel("Monto total smurf")
    axis.legend(title="Riesgo máximo", loc="best")
    return axis


def plot_q13_bad_loans_with_frequency(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica pares con préstamos incobrables y alta frecuencia."""
    data = question13_bad_loans_with_frequency(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q13 – Préstamos y frecuencia",
            "Sin coincidencias de préstamos incobrables.",
        )

    work = data.copy()
    work["pair"] = work["pair"].fillna("sin_par").astype(str)
    work = work.sort_values(
        ["meses_con_coincidencia", "monto_prestamos_incumplidos"],
        ascending=[False, False],
    ).head(top_n)
    sns.barplot(
        data=work,
        x="monto_prestamos_incumplidos",
        y="pair",
        hue="eventos_alta_frecuencia",
        ax=axis,
    )
    axis.set_title(f"Q13 – Préstamos incobrables frecuentes ({timeframe})")
    axis.set_xlabel("Monto en préstamos incumplidos")
    axis.set_ylabel("Par emisor → receptor")
    axis.legend(title="Eventos alta frecuencia")
    return axis


def plot_q14_recurrent_payroll(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica pagos recurrentes tipo nómina."""
    data = question14_recurrent_payroll(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _empty_chart(
            axis,
            "Q14 – Pagos recurrentes",
            "Sin patrones de nómina recurrente.",
        )

    work = data.copy()
    work["emisor"] = work["emisor"].fillna("sin_emisor").astype(str)
    work["receptor"] = work["receptor"].fillna("sin_receptor").astype(str)
    work["pair"] = work["emisor"] + "→" + work["receptor"]
    work = work.sort_values(
        ["meses_recurrentes", "monto_total"],
        ascending=[False, False],
    ).head(top_n)
    sns.barplot(
        data=work,
        x="monto_total",
        y="pair",
        hue="meses_recurrentes",
        ax=axis,
    )
    axis.set_title(f"Q14 – Pagos recurrentes tipo nómina ({timeframe})")
    axis.set_xlabel("Monto total pagado")
    axis.set_ylabel("Emisor → Receptor")
    axis.legend(title="Meses recurrentes")
    return axis


def plot_q15_coordinated_cluster_signals(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 8,
) -> Axes:
    """Grafica los clusters con mayor número de señales coordinadas."""

    data = question15_coordinated_cluster_signals(reports, timeframe, top_n=top_n)
    axis = _ensure_axis(ax, figsize=(10, 6))

    cluster_series = data.get("cluster_id") if "cluster_id" in data else pd.Series(dtype=str)
    if data.empty or (not cluster_series.empty and cluster_series.eq("sin_datos").all()):
        return _empty_chart(
            axis,
            "Q15 – Clusters coordinados",
            "Sin clusters con señales coordinadas en el periodo seleccionado.",
        )

    signal_cols = {
        "yo_yo_cluster_tasa_flag": "Yo-Yo",
        "smurf_cluster_tasa_flag": "Smurf",
        "red_cluster_tasa_en_ciclos": "Ciclos",
        "quid_cluster_tasa_flag": "Quid",
        "referencia_cluster_tasa_reutilizada": "Referencia",
    }

    work = data.copy()
    if "cluster_id" not in work:
        work["cluster_id"] = "cluster_sin_id"
    work["cluster_id"] = work["cluster_id"].fillna("cluster_sin_id").astype(str)
    work = work.sort_values(
        ["signals_activas", "riesgo_cluster_maximo", "cluster_tx_sum"],
        ascending=[False, False, False],
    ).head(top_n)

    heatmap_df = (
        work.set_index("cluster_id")[list(signal_cols.keys())]
        .fillna(0.0)
        .astype(float)
        .rename(columns=signal_cols)
        * 100
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".0f",
        cmap="OrRd",
        ax=axis,
        cbar_kws={"label": "% de transacciones con señal"},
    )
    axis.set_title(f"Q15 – Señales coordinadas por cluster ({timeframe})")
    axis.set_xlabel("Señal priorizada")
    axis.set_ylabel("Cluster de personas")
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha="right")
    return axis


def plot_q16_multisignal_transactions(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 25,
) -> Axes:
    """Grafica las transacciones con mayor número de señales simultáneas."""

    data = question16_multisignal_transactions(reports, timeframe, top_n=top_n)
    axis = _ensure_axis(ax, figsize=(10, 6))

    if data.empty or data.get("signals_activas", pd.Series(dtype=int)).max() == 0:
        return _empty_chart(
            axis,
            "Q16 – Transacciones multi-señal",
            "Sin transacciones con múltiples señales en el periodo seleccionado.",
        )

    work = data.copy()
    work["movement_amount"] = pd.to_numeric(work.get("movement_amount", 0.0), errors="coerce").fillna(0.0)
    work["risk_score"] = pd.to_numeric(work.get("risk_score", 0.0), errors="coerce").fillna(0.0)
    work["signals_activas"] = (
        pd.to_numeric(work.get("signals_activas", 0), errors="coerce").fillna(0).astype(int)
    )
    if "flag_jerarquia" not in work:
        work["flag_jerarquia"] = False
    work["flag_jerarquia"] = work["flag_jerarquia"].fillna(False).astype(bool)
    work["relacion_label"] = work["flag_jerarquia"].map({True: "Jerárquica", False: "No jerárquica"})
    if COL_SENDER_ID not in work:
        work[COL_SENDER_ID] = "sin_emisor"
    if COL_RECEIVER_ID not in work:
        work[COL_RECEIVER_ID] = "sin_receptor"
    work["emisor"] = work[COL_SENDER_ID].fillna("sin_emisor").astype(str)
    work["receptor"] = work[COL_RECEIVER_ID].fillna("sin_receptor").astype(str)
    work["pair"] = work["emisor"] + "→" + work["receptor"]
    work = work.sort_values(
        ["signals_activas", "risk_score", "movement_amount"],
        ascending=[False, False, False],
    ).head(top_n)

    scatter = sns.scatterplot(
        data=work,
        x="movement_amount",
        y="risk_score",
        hue="signals_activas",
        size="signals_activas",
        style="relacion_label",
        palette="viridis",
        sizes=(60, 280),
        legend="brief",
        ax=axis,
    )
    for _, row in work.iterrows():
        scatter.text(
            row["movement_amount"],
            row["risk_score"] + 0.02,
            row.get("pair", "sin_par"),
            fontsize=8,
            ha="left",
        )

    axis.set_title(f"Q16 – Transacciones con múltiples señales ({timeframe})")
    axis.set_xlabel("Monto transaccionado")
    axis.set_ylabel("Riesgo (risk_score)")
    axis.legend(title="Señales activas", loc="best")
    return axis


def plot_q17_nlp_person_profiles(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 15,
) -> Axes:
    """Grafica las personas con más señales NLP y su riesgo promedio."""

    data = question17_nlp_person_profiles(reports, timeframe, top_n=top_n)
    axis = _ensure_axis(ax, figsize=(10, 6))

    persona_series = data.get("persona") if "persona" in data else pd.Series(dtype=str)
    if data.empty or (not persona_series.empty and persona_series.eq("sin_persona").all()):
        return _empty_chart(
            axis,
            "Q17 – Perfiles NLP",
            "Sin personas con conceptos NLP sospechosos en el periodo.",
        )

    work = data.copy()
    if "persona" not in work:
        work["persona"] = "sin_persona"
    work["persona"] = work["persona"].fillna("sin_persona").astype(str)
    work["tx_sospechosas_nlp"] = (
        pd.to_numeric(work.get("tx_sospechosas_nlp", 0), errors="coerce").fillna(0).astype(int)
    )
    work["conceptos_unicos"] = (
        pd.to_numeric(work.get("conceptos_unicos", 0), errors="coerce").fillna(0).astype(int)
    )
    work["risk_avg_person"] = (
        pd.to_numeric(work.get("risk_avg_person", 0.0), errors="coerce").fillna(0.0).astype(float)
    )
    work["proporcion_sospechosa"] = (
        pd.to_numeric(work.get("proporcion_sospechosa", 0.0), errors="coerce").fillna(0.0)
    )
    work["top_conceptos_display"] = (
        work.get("top_conceptos_display", "sin_top_conceptos")
        .fillna("sin_top_conceptos")
        .astype(str)
    )
    work = work.sort_values(
        ["tx_sospechosas_nlp", "proporcion_sospechosa", "risk_avg_person"],
        ascending=[False, False, False],
    ).head(top_n)

    bins = [-float("inf"), 1.0, 2.0, 3.0, float("inf")]
    labels = ["≤1.0", "1.0–2.0", "2.0–3.0", ">3.0"]
    work["riesgo_categoria"] = pd.cut(
        work["risk_avg_person"], bins=bins, labels=labels, include_lowest=True, right=False
    )

    scatter = sns.scatterplot(
        data=work,
        x="tx_sospechosas_nlp",
        y="persona",
        hue="riesgo_categoria",
        size="conceptos_unicos",
        palette="magma",
        sizes=(60, 280),
        legend="brief",
        ax=axis,
    )

    for _, row in work.iterrows():
        scatter.text(
            row["tx_sospechosas_nlp"] + 0.1,
            row["persona"],
            f"{row.get('top_conceptos_display', 'sin_top_conceptos')} ({row.get('proporcion_sospechosa', 0):.0%})",
            fontsize=8,
            va="center",
        )

    axis.set_title(f"Q17 – Perfiles NLP sospechosos ({timeframe})")
    axis.set_xlabel("Transacciones NLP sospechosas")
    axis.set_ylabel("Persona")
    axis.legend(title="Riesgo promedio")
    return axis


__all__ = [
    "plot_q1_manager_nlp",
    "plot_q2_manager_concepts",
    "plot_q3_quid_pairs",
    "plot_q4_negative_value_vs_load",
    "plot_q5_reference_reuse",
    "plot_q6_centralizers",
    "plot_q7_net_imbalance",
    "plot_q8_case13_new_employees",
    "plot_q9_case14_veterans_from_newcomers",
    "plot_q10_yoyo_streaks",
    "plot_q11_near_threshold_structuring",
    "plot_q12_smurfing_chronic",
    "plot_q13_bad_loans_with_frequency",
    "plot_q14_recurrent_payroll",
    "plot_q15_coordinated_cluster_signals",
    "plot_q16_multisignal_transactions",
    "plot_q17_nlp_person_profiles",
]
