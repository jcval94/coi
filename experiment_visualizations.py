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
]
