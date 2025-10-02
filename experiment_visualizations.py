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


def plot_q15_coordinated_cluster_signals(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica clusters resaltando el número de señales coordinadas activas."""

    data = question15_coordinated_cluster_signals(reports, timeframe)
    axis = _ensure_axis(ax, figsize=(10, 6))
    if data.empty or "cluster_id" not in data:
        return _empty_chart(
            axis,
            "Q15 – Señales coordinadas por cluster",
            "Sin clusters relevantes para el periodo.",
        )

    work = data.copy()
    if "signals_activas" not in work:
        return _empty_chart(
            axis,
            "Q15 – Señales coordinadas por cluster",
            "El resultado no incluye el total de señales activas.",
        )

    work["cluster_label"] = work["cluster_id"].fillna("cluster_sin_id").astype(str)
    work = work.sort_values(
        ["signals_activas", "riesgo_cluster_maximo", "cluster_tx_sum"],
        ascending=[False, False, False],
    ).head(top_n)

    sns.barplot(
        data=work,
        x="signals_activas",
        y="cluster_label",
        ax=axis,
        palette="Blues_d",
    )
    axis.set_title(f"Q15 – Clusters con mayor coordinación ({timeframe})")
    axis.set_xlabel("Número de señales priorizadas activas")
    axis.set_ylabel("Cluster")

    if hasattr(axis, "bar_label") and axis.containers:
        axis.bar_label(
            axis.containers[0],
            labels=[f"riesgo máx {row:.2f}" for row in work["riesgo_cluster_maximo"]],
            padding=6,
        )

    return axis


def plot_q16_multisignal_transactions(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 20,
) -> Axes:
    """Grafica las transacciones con múltiples señales simultáneas."""

    data = question16_multisignal_transactions(reports, timeframe)
    axis = _ensure_axis(ax, figsize=(11, 7))
    if data.empty or "risk_score" not in data:
        return _empty_chart(
            axis,
            "Q16 – Transacciones multisignales",
            "Sin transacciones destacadas para el periodo.",
        )

    work = data.copy()
    if {"emisor", "receptor"}.issubset(work.columns):
        work["tx_label"] = work.apply(
            lambda row: (
                f"{row.get('fecha_hora_ts', 'sin_fecha')} "
                f"{row.get('emisor', 'sin_emisor')}→{row.get('receptor', 'sin_receptor')}"
            ),
            axis=1,
        )
    else:
        work["tx_label"] = work.index.astype(str)

    work = work.sort_values(
        ["signals_activas", "risk_score", "movement_amount"],
        ascending=[False, False, False],
    ).head(top_n)

    sns.barplot(
        data=work,
        x="risk_score",
        y="tx_label",
        hue="signals_activas",
        ax=axis,
        palette="rocket",
    )
    axis.set_title(f"Q16 – Transacciones con señales simultáneas ({timeframe})")
    axis.set_xlabel("risk_score")
    axis.set_ylabel("Transacción (fecha emisor→receptor)")
    axis.legend(title="Señales activas", bbox_to_anchor=(1.02, 1), loc="upper left")
    return axis


def plot_q17_nlp_person_profiles(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 15,
) -> Axes:
    """Grafica las personas con mayor concentración de transacciones sospechosas por NLP."""

    data = question17_nlp_person_profiles(reports, timeframe)
    axis = _ensure_axis(ax, figsize=(10, 6))
    if data.empty or "tx_sospechosas_nlp" not in data:
        return _empty_chart(
            axis,
            "Q17 – Personas NLP prioritarias",
            "Sin personas con señales NLP para el periodo.",
        )

    work = data.copy()
    work["persona"] = work["persona"].fillna("sin_persona").astype(str)
    work = work.sort_values(
        ["tx_sospechosas_nlp", "proporcion_sospechosa", "risk_avg_person"],
        ascending=[False, False, False],
    ).head(top_n)

    sns.barplot(
        data=work,
        x="tx_sospechosas_nlp",
        y="persona",
        hue="conceptos_unicos",
        ax=axis,
        palette="mako",
    )
    axis.set_title(f"Q17 – Personas con más actividad NLP sospechosa ({timeframe})")
    axis.set_xlabel("Transacciones NLP sospechosas")
    axis.set_ylabel("Persona")
    axis.legend(title="Conceptos únicos", bbox_to_anchor=(1.02, 1), loc="upper left")
    return axis


__all__ = [
    "plot_q1_manager_nlp",
    "plot_q2_manager_concepts",
    "plot_q3_quid_pairs",
    "plot_q4_negative_value_vs_load",
    "plot_q5_reference_reuse",
    "plot_q6_centralizers",
    "plot_q7_net_imbalance",
    "plot_q15_coordinated_cluster_signals",
    "plot_q16_multisignal_transactions",
    "plot_q17_nlp_person_profiles",
]
