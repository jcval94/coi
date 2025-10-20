"""Visualizaciones con Seaborn para las preguntas de `experiment_questions`."""
from __future__ import annotations

from typing import Any, Mapping, Optional
from textwrap import fill

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.axes import Axes

from coi_fraud.schemas import COL_RECEIVER_ID, COL_SENDER_ID
from experiment_questions import (
    DEFAULT_TIMEFRAME,
    QUESTION1_DIRECTIONS,
    QUESTION_METADATA,
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
    question18_user_risk_scores,
)


sns.set_theme(style="whitegrid")


LEGEND_OUTSIDE_DEFAULT_KWARGS = {
    "loc": "center left",
    "bbox_to_anchor": (1.02, 0.5),
    "borderaxespad": 0.0,
    "frameon": False,
}


TIMEFRAME_LABELS = {
    "ultimo_mes": "Último mes",
    "ultimos_3_meses": "Últimos 3 meses",
    DEFAULT_TIMEFRAME: "Todo el tiempo",
}


def _ensure_axis(ax: Optional[Axes] = None, figsize: tuple[int, int] = (8, 5)) -> Axes:
    if ax is not None:
        return ax
    _, created_ax = plt.subplots(figsize=figsize)
    return created_ax


def _format_timeframe_label(timeframe: str) -> str:
    return TIMEFRAME_LABELS.get(timeframe, timeframe.replace("_", " ").capitalize())


def _apply_plot_metadata(ax: Axes, question_key: str, timeframe: str) -> None:
    meta = QUESTION_METADATA.get(question_key, {})
    title = meta.get("title", question_key)
    timeframe_label = _format_timeframe_label(timeframe)
    description = meta.get("description")
    if description:
        subtitle = fill(description, width=90)
        ax.set_title(f"{title} · {timeframe_label}\n{subtitle}", loc="left", fontsize=12)
    else:
        ax.set_title(f"{title} · {timeframe_label}", loc="left", fontsize=12)


def _render_empty_chart(ax: Axes, question_key: str, timeframe: str, message: str) -> Axes:
    _apply_plot_metadata(ax, question_key, timeframe)
    ax.text(
        0.5,
        0.5,
        fill(message, width=80),
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        color="#555555",
    )
    ax.set_axis_off()
    return ax


def _legend_outside(
    axis: Axes,
    *args: Any,
    adjust_right: Optional[float] = 0.78,
    add_artist: bool = False,
    **kwargs: Any,
) -> Optional[Any]:
    """Place the legend outside of the plotting area with sensible defaults."""

    if not args and axis.get_legend_handles_labels() == ([], []):
        return None

    legend_kwargs = {**LEGEND_OUTSIDE_DEFAULT_KWARGS, **kwargs}
    legend = axis.legend(*args, **legend_kwargs)
    if add_artist:
        axis.add_artist(legend)

    if adjust_right is not None and axis.figure is not None:
        current_right = axis.figure.subplotpars.right
        if current_right > adjust_right:
            axis.figure.subplots_adjust(right=adjust_right)

    return legend


def plot_q1_manager_nlp(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
    direction: str = "manager_a_subordinado",
    invert_direction: bool = False,
) -> Axes:
    """Grafica los pares manager-subordinado con conceptos NLP sospechosos.

    Parameters
    ----------
    reports
        Salidas tabulares del pipeline de detección de fraude.
    timeframe
        Ventana temporal a consultar.
    ax
        Eje de Matplotlib donde dibujar la gráfica.
    top_n
        Número máximo de pares a mostrar.
    direction
        Dirección inicial del flujo de pagos a visualizar. Acepta
        ``"manager_a_subordinado"`` o ``"subordinado_a_manager"``.
    invert_direction
        Cuando es ``True`` invierte el flujo indicado en ``direction`` para
        permitir la visualización del sentido contrario sin necesidad de
        recalcular la tabla previa.
    """
    direction = str(direction)
    if direction not in QUESTION1_DIRECTIONS:
        valid = "', '".join(QUESTION1_DIRECTIONS)
        raise ValueError(
            "direction debe ser uno de '{valid}', se recibió '{direction}'".format(
                valid=valid, direction=direction
            )
        )
    if invert_direction:
        direction = (
            "subordinado_a_manager"
            if direction == "manager_a_subordinado"
            else "manager_a_subordinado"
        )

    data = question1_manager_nlp(reports, timeframe, direction=direction)
    axis = _ensure_axis(ax)
    if data.empty:
        return _render_empty_chart(
            axis,
            "q1_manager_nlp",
            timeframe,
            "Sin coincidencias manager-subordinado para el periodo seleccionado.",
        )

    work = data.copy()
    work["concepto_label"] = work["nlp_concepto_sospechoso"].apply(
        lambda values: ", ".join(values) if isinstance(values, list) and values else "SIN_CONCEPTO"
    )
    if direction == "manager_a_subordinado":
        arrow = "→"
        y_label = "Manager → Subordinado"
        left = work["manager_user_id"].fillna("sin_manager").astype(str)
        right = work["subordinado_user_id"].fillna("sin_subordinado").astype(str)
    elif direction == "subordinado_a_manager":
        arrow = "→"
        y_label = "Subordinado → Manager"
        left = work["subordinado_user_id"].fillna("sin_subordinado").astype(str)
        right = work["manager_user_id"].fillna("sin_manager").astype(str)
    else:
        valid = "', '".join(QUESTION1_DIRECTIONS)
        raise ValueError(
            "direction debe ser uno de '{valid}', se recibió '{direction}'".format(
                valid=valid, direction=direction
            )
        )

    work["pair"] = left + arrow + right
    work = work.sort_values(["tx_count", "monto_total"], ascending=[False, False]).head(top_n)

    sns.barplot(data=work, x="tx_count", y="pair", hue="concepto_label", ax=axis)
    _apply_plot_metadata(axis, "q1_manager_nlp", timeframe)
    axis.set_xlabel("Número de transacciones")
    axis.set_ylabel(y_label)
    if axis.get_legend() is not None:
        handles, labels = axis.get_legend_handles_labels()
        _legend_outside(axis, handles, labels, title="Concepto")
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
        return _render_empty_chart(
            axis,
            "q2_manager_concepts",
            timeframe,
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
        _legend_outside(axis, title="Mes")
    _apply_plot_metadata(axis, "q2_manager_concepts", timeframe)
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
    """Grafica pares destacados por puntaje de "algo por algo"."""
    data = question3_quid_pairs(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _render_empty_chart(
            axis,
            "q3_quid_pairs",
            timeframe,
            "Sin pares destacados para el periodo.",
        )

    work = data.copy()
    if "nivel_respuesta" in work.columns:
        work = work.loc[
            work["nivel_respuesta"] == "resumen_del_par_algo_por_algo"
        ].copy()
    if work.empty:
        return _render_empty_chart(
            axis,
            "q3_quid_pairs",
            timeframe,
            "Solo se encontraron detalles de transacción, no pares agregados.",
        )

    work["label"] = (
        work.get(
            "resumen_personas_involucradas",
            work.get("identificador_emisor_a_receptor", "sin_par"),
        )
        .fillna("sin_par")
        .astype(str)
    )
    work = work.sort_values(
        [
            "puntaje_algo_por_algo_mas_alto_en_el_par",
            "cantidad_movimientos_con_indicio_de_algo_por_algo",
        ],
        ascending=[False, False],
    ).head(top_n)
    sns.barplot(
        data=work,
        x="puntaje_algo_por_algo_mas_alto_en_el_par",
        y="label",
        hue="cantidad_movimientos_con_indicio_de_algo_por_algo",
        ax=axis,
        palette="Reds",
    )
    _apply_plot_metadata(axis, "q3_quid_pairs", timeframe)
    axis.set_xlabel("Puntaje más alto de 'algo por algo'")
    axis.set_ylabel("Par emisor → receptor")
    _legend_outside(axis, title="Movimientos detectados")
    return axis


def plot_q3_algo_pair_detalle(
    record: Mapping[str, Any] | pd.Series,
    *,
    ax: Optional[Axes] = None,
) -> Axes:
    """Grafica un resumen visual de un solo par "algo por algo"."""

    axis = _ensure_axis(ax, figsize=(7, 4))
    series = pd.Series(record)

    def _as_float(value: Any) -> float:
        try:
            if value is None:
                return 0.0
            if isinstance(value, float) and pd.isna(value):
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    label = (
        series.get("resumen_personas_involucradas")
        or series.get("identificador_emisor_a_receptor")
    )
    label = str(label) if label else "Par sin nombre"
    metrics = pd.DataFrame(
        {
            "factor": [
                "Movimientos sospechosos",
                "Jefes involucrados (%)",
                "Textos con aprobación (%)",
                "Textos con compensación (%)",
                "Puntaje más alto",
                "Riesgo máximo",
            ],
            "valor": [
                _as_float(
                    series.get("cantidad_movimientos_con_indicio_de_algo_por_algo")
                ),
                _as_float(
                    series.get("porcentaje_movimientos_donde_participa_un_jefe")
                ),
                _as_float(
                    series.get("porcentaje_movimientos_con_texto_de_aprobacion")
                ),
                _as_float(
                    series.get("porcentaje_movimientos_con_texto_de_compensacion")
                ),
                _as_float(series.get("puntaje_algo_por_algo_mas_alto_en_el_par")),
                _as_float(series.get("riesgo_maximo_de_los_movimientos_relacionados")),
            ],
        }
    )

    sns.barplot(data=metrics, x="valor", y="factor", palette="Reds", ax=axis)
    max_value = float(metrics["valor"].max()) if not metrics.empty else 0.0
    offset = max(max_value * 0.02, 0.2)
    for index, value in enumerate(metrics["valor"]):
        axis.text(value + offset, index, f"{value:.1f}", va="center")

    axis.set_title(f"¿Por qué preocupa {label}?", loc="left")
    axis.set_xlabel("Valor (conteo o porcentaje)")
    axis.set_ylabel("")
    axis.grid(True, axis="x", linestyle="--", alpha=0.3)
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
        return _render_empty_chart(
            axis,
            "q4_quid_negative_value_vs_load",
            timeframe,
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
        _legend_outside(axis, title="Puntaje")
    _apply_plot_metadata(axis, "q4_quid_negative_value_vs_load", timeframe)
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
    """Grafica receptores que reciben conceptos sospechosos desde múltiples emisores."""
    data = question5_reference_reuse(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _render_empty_chart(
            axis,
            "q5_reference_reuse",
            timeframe,
            "Sin receptores con conceptos sospechosos repetidos por múltiples emisores.",
        )

    work = data.copy()
    if "nivel_respuesta" in work.columns:
        work = work.loc[work["nivel_respuesta"] == "concepto_receptor"].copy()
    if work.empty:
        return _render_empty_chart(
            axis,
            "q5_reference_reuse",
            timeframe,
            "No se encontraron combinaciones receptor-concepto multi-emisor.",
        )

    work = work.sort_values(
        ["monto_total", "emisores_unicos", "tx_count"],
        ascending=[False, False, False],
    ).head(top_n)
    work["label"] = (
        work.get(COL_RECEIVER_ID, pd.Series(dtype="object")).fillna("sin_receptor").astype(str)
        + " ← "
        + work.get("nlp_concepto_sospechoso", pd.Series(dtype="object")).fillna("SIN_CONCEPTO").astype(str)
    )

    cmap = sns.color_palette("viridis", as_cmap=True)
    tx_counts = work["tx_count"].astype(float)
    vmin = (tx_counts.min() - 0.5) if tx_counts.size else 0
    vmax = (tx_counts.max() + 0.5) if tx_counts.size else 1
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    bar_colors = cmap(norm(tx_counts))
    bars = axis.barh(work["label"], work["monto_total"], color=bar_colors)

    max_total = work["monto_total"].max() if not work["monto_total"].empty else 0
    axis.set_xlim(0, max_total * 1.1 if max_total > 0 else 1)
    for bar, amount in zip(bars, work["monto_total"]):
        width = bar.get_width()
        axis.text(
            width,
            bar.get_y() + bar.get_height() / 2,
            f" {amount:,.2f}",
            va="center",
            ha="left",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = axis.figure.colorbar(sm, ax=axis)
    colorbar.set_label("Número de transacciones")

    _apply_plot_metadata(axis, "q5_reference_reuse", timeframe)
    axis.set_xlabel("Monto total")
    axis.set_ylabel("Receptor ← concepto")
    axis.invert_yaxis()
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
        return _render_empty_chart(
            axis,
            "q6_centralizers",
            timeframe,
            "Sin receptores centralizadores para el periodo.",
        )

    work = data.copy()
    if "month_id" not in work.columns:
        return _render_empty_chart(
            axis,
            "q6_centralizers",
            timeframe,
            "Sin información temporal para graficar centralidad.",
        )

    top_receivers = (
        work.groupby(COL_RECEIVER_ID, observed=True)["centralidad"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    work = work.loc[work[COL_RECEIVER_ID].isin(top_receivers)].copy()
    if work.empty:
        return _render_empty_chart(
            axis,
            "q6_centralizers",
            timeframe,
            "Sin receptores centralizadores suficientes para graficar.",
        )

    work[COL_RECEIVER_ID] = work[COL_RECEIVER_ID].fillna("sin_receptor").astype(str)
    work = work.sort_values(["month_id", COL_RECEIVER_ID])
    sns.lineplot(
        data=work,
        x="month_id",
        y="centralidad",
        hue=COL_RECEIVER_ID,
        style=COL_RECEIVER_ID,
        markers=True,
        dashes=False,
        ax=axis,
    )
    if axis.get_legend() is not None:
        _legend_outside(axis, title="Receptor")
    _apply_plot_metadata(axis, "q6_centralizers", timeframe)
    axis.set_xlabel("Mes")
    axis.set_ylabel("Centralidad (inflow × emisores únicos)")
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
        return _render_empty_chart(
            axis,
            "q7_net_imbalance",
            timeframe,
            "Sin personas con desbalance calculado.",
        )

    work = data.copy()
    work["persona"] = work["persona"].fillna("sin_persona").astype(str)
    work["abs_neto"] = work["desbalance_persona_monto_neto"].abs()
    work = work.sort_values("abs_neto", ascending=False).head(top_n)
    sns.barplot(data=work, x="desbalance_persona_monto_neto", y="persona", ax=axis)
    _apply_plot_metadata(axis, "q7_net_imbalance", timeframe)
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
        return _render_empty_chart(
            axis,
            "q8_case13_new_employees",
            timeframe,
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
    _apply_plot_metadata(axis, "q8_case13_new_employees", timeframe)
    axis.set_xlabel("Monto total recibido")
    axis.set_ylabel("Receptor")
    _legend_outside(axis, title="Tx monto alto")
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
        return _render_empty_chart(
            axis,
            "q9_case14_veterans_from_newcomers",
            timeframe,
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
    _apply_plot_metadata(axis, "q9_case14_veterans_from_newcomers", timeframe)
    axis.set_xlabel("Monto recibido desde emisores nuevos")
    axis.set_ylabel("Receptor veterano")
    _legend_outside(axis, title="Emisores únicos")
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
    axis = _ensure_axis(ax, figsize=(10, max(5, int(top_n * 0.7))))
    if data.empty:
        return _render_empty_chart(
            axis,
            "q10_yoyo_streaks",
            timeframe,
            "Sin rachas Yo-Yo identificadas.",
        )

    work = data.copy()
    work["par_bidir"] = work["par_bidir"].fillna("sin_par").astype(str)
    work = work.sort_values(
        ["racha_max_yo_yo", "riesgo_max_par", "tx_yo_yo_totales"],
        ascending=[False, False, False],
    ).head(top_n)
    ordered_pairs = work["par_bidir"].tolist()
    work["par_bidir"] = pd.Categorical(work["par_bidir"], categories=ordered_pairs, ordered=True)

    palette = sns.color_palette("rocket", as_cmap=True)
    sns.scatterplot(
        data=work,
        x="racha_max_yo_yo",
        y="par_bidir",
        hue="riesgo_max_par",
        size="tx_yo_yo_totales",
        palette=palette,
        sizes=(70, 600),
        linewidth=0.6,
        edgecolor="#333333",
        legend=False,
        ax=axis,
    )

    riesgo_min = float(work["riesgo_max_par"].min())
    riesgo_max = float(work["riesgo_max_par"].max())
    if np.isclose(riesgo_min, riesgo_max):
        riesgo_max = riesgo_min + 1.0
    risk_norm = colors.Normalize(vmin=riesgo_min, vmax=riesgo_max)
    colorbar = axis.figure.colorbar(
        plt.cm.ScalarMappable(norm=risk_norm, cmap=palette),
        ax=axis,
        pad=0.01,
    )
    colorbar.set_label("Riesgo máximo del par")

    size_min = float(work["tx_yo_yo_totales"].min())
    size_max = float(work["tx_yo_yo_totales"].max())
    size_range = (70.0, 600.0)

    def _scale_size(value: float) -> float:
        if size_min == size_max:
            return float(np.mean(size_range))
        return float(np.interp(value, (size_min, size_max), size_range))

    size_ticks = np.linspace(size_min, size_max, num=min(4, len(work)))
    size_handles = [
        plt.scatter([], [], s=_scale_size(val), color="#555555", alpha=0.6)
        for val in size_ticks
    ]
    size_labels = [f"{int(round(val))} tx" for val in size_ticks]
    if size_handles:
        _legend_outside(
            size_handles,
            size_labels,
            title="Transacciones Yo-Yo",
            loc="center left",
            bbox_to_anchor=(1.02, 0.2),
            frameon=True,
            adjust_right=0.8,
            add_artist=True,
        )

    racha_min = float(work["racha_max_yo_yo"].min())
    racha_max = float(work["racha_max_yo_yo"].max())
    x_offset = max((racha_max - racha_min) * 0.05, 0.5)
    for _, row in work.iterrows():
        meses = row.get("meses_con_yo_yo")
        if pd.isna(meses):
            label = "sin meses"
        else:
            meses_int = int(meses)
            label = f"{meses_int} mes" if meses_int == 1 else f"{meses_int} meses"
        axis.text(
            row["racha_max_yo_yo"] + x_offset,
            row["par_bidir"],
            label,
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
        )

    axis.set_xlim(left=0)
    axis.margins(x=0.05)
    axis.grid(axis="x", which="major", linestyle="--", alpha=0.3)
    _apply_plot_metadata(axis, "q10_yoyo_streaks", timeframe)
    axis.set_xlabel("Racha máxima Yo-Yo")
    axis.set_ylabel("Par bidireccional")
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
        return _render_empty_chart(
            axis,
            "q11_near_threshold_structuring",
            timeframe,
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
    _apply_plot_metadata(axis, "q11_near_threshold_structuring", timeframe)
    axis.set_xlabel("Meses con montos cercanos")
    axis.set_ylabel("Par emisor → receptor")
    _legend_outside(axis, title="Riesgo máximo")
    return axis


def plot_q12_smurfing_chronic(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 10,
) -> Axes:
    """Grafica pares con patrones crónicos de fraccionamiento."""
    data = question12_smurfing_chronic(reports, timeframe)
    axis = _ensure_axis(ax)
    if data.empty:
        return _render_empty_chart(
            axis,
            "q12_smurfing_chronic",
            timeframe,
            "Sin pares con fraccionamiento prolongado.",
        )

    work = data.copy()
    work["pair"] = work["pair"].fillna("sin_par").astype(str)
    work = work.sort_values(
        ["monto_fraccionado_total", "transacciones_fraccionadas"],
        ascending=[False, False],
    ).head(top_n)
    sns.scatterplot(
        data=work,
        x="monto_fraccionado_total",
        y="pair",
        size="transacciones_fraccionadas",
        hue="riesgo_maximo",
        palette="viridis",
        ax=axis,
    )
    _apply_plot_metadata(axis, "q12_smurfing_chronic", timeframe)
    axis.set_xlabel("Monto total fraccionado")
    axis.set_ylabel("Par emisor → receptor")
    _legend_outside(axis, title="Riesgo máximo")
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
        return _render_empty_chart(
            axis,
            "q13_bad_loans_with_frequency",
            timeframe,
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
    _apply_plot_metadata(axis, "q13_bad_loans_with_frequency", timeframe)
    axis.set_xlabel("Monto en préstamos incumplidos")
    axis.set_ylabel("Par emisor → receptor")
    _legend_outside(axis, title="Eventos alta frecuencia")
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
        return _render_empty_chart(
            axis,
            "q14_recurrent_payroll",
            timeframe,
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
    _apply_plot_metadata(axis, "q14_recurrent_payroll", timeframe)
    axis.set_xlabel("Monto total pagado")
    axis.set_ylabel("Emisor → Receptor")
    _legend_outside(axis, title="Meses recurrentes")
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
        return _render_empty_chart(
            axis,
            "q15_coordinated_cluster_signals",
            timeframe,
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

    display_df = heatmap_df.replace(0.0, float("nan"))
    mask = display_df.isna()
    annotations = heatmap_df.where(~mask)
    if display_df.empty or display_df.isna().all().all():
        vmax = 1.0
    else:
        vmax = float(np.nanmax(display_df.values))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
    norm = colors.PowerNorm(gamma=0.6, vmin=0.0, vmax=vmax)

    sns.heatmap(
        display_df,
        annot=annotations,
        fmt=".0f",
        cmap="YlOrRd",
        norm=norm,
        linewidths=0.6,
        linecolor="#f0f0f0",
        mask=mask,
        ax=axis,
        cbar_kws={"label": "% de transacciones con señal"},
    )
    _apply_plot_metadata(axis, "q15_coordinated_cluster_signals", timeframe)
    axis.set_xlabel("Señal priorizada")
    axis.set_ylabel("Cluster de personas")
    axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha="right")
    axis.set_xlim(-0.5, len(signal_cols) - 0.5 + 1.6)

    for idx, (_, row) in enumerate(work.iterrows()):
        axis.text(
            len(signal_cols) + 0.35,
            idx + 0.5,
            (
                f"{int(row.get('signals_activas', 0))} señales · "
                f"riesgo {row.get('riesgo_cluster_maximo', 0):.2f} · "
                f"{int(row.get('cluster_tx_count', 0))} tx"
            ),
            va="center",
            ha="left",
            fontsize=10,
            color="#333333",
        )

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
        return _render_empty_chart(
            axis,
            "q16_multisignal_transactions",
            timeframe,
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

    _apply_plot_metadata(axis, "q16_multisignal_transactions", timeframe)
    axis.set_xlabel("Monto transaccionado")
    axis.set_ylabel("Riesgo (risk_score)")
    _legend_outside(axis, title="Señales activas")
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
        return _render_empty_chart(
            axis,
            "q17_nlp_person_profiles",
            timeframe,
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
    work["score_probable_coi"] = (
        pd.to_numeric(work.get("score_probable_coi", 0.0), errors="coerce").fillna(0.0)
    )
    work["sentimiento_promedio"] = (
        pd.to_numeric(work.get("sentimiento_promedio", 0.0), errors="coerce").fillna(0.0)
    )
    if "sentimiento_etiqueta" not in work:
        work["sentimiento_etiqueta"] = "neutral"
    work["top_conceptos_display"] = (
        work.get("top_conceptos_display", "sin_top_conceptos")
        .fillna("sin_top_conceptos")
        .astype(str)
    )
    work = work.sort_values(
        ["score_probable_coi", "tx_sospechosas_nlp", "proporcion_sospechosa", "risk_avg_person"],
        ascending=[False, False, False, False],
    ).head(top_n)

    bins = [-float("inf"), 1.0, 2.0, 3.0, float("inf")]
    labels = ["≤1.0", "1.0–2.0", "2.0–3.0", ">3.0"]
    work["riesgo_categoria"] = pd.cut(
        work["risk_avg_person"], bins=bins, labels=labels, include_lowest=True, right=False
    )

    scatter = sns.scatterplot(
        data=work,
        x="score_probable_coi",
        y="persona",
        hue="riesgo_categoria",
        size="tx_sospechosas_nlp",
        palette="magma",
        sizes=(60, 280),
        legend="brief",
        ax=axis,
    )

    for _, row in work.iterrows():
        scatter.text(
            row["score_probable_coi"] + 0.05,
            row["persona"],
            (
                f"{row.get('top_conceptos_display', 'sin_top_conceptos')} "
                f"(tx={int(row.get('tx_sospechosas_nlp', 0))}, score={row.get('score_probable_coi', 0):.2f}, "
                f"sent={row.get('sentimiento_etiqueta', 'neutral')})"
            ),
            fontsize=8,
            va="center",
        )

    _apply_plot_metadata(axis, "q17_nlp_person_profiles", timeframe)
    axis.set_xlabel("Score probable COI (mayor es más riesgoso)")
    axis.set_ylabel("Persona")
    _legend_outside(axis, title="Riesgo promedio")
    return axis


def plot_q18_user_risk_scores(
    reports: dict,
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    ax: Optional[Axes] = None,
    top_n: int = 15,
) -> Axes:
    """Grafica las personas priorizadas por riesgo promedio y desbalance."""

    data = question18_user_risk_scores(reports, timeframe, top_n=top_n)
    axis = _ensure_axis(ax, figsize=(10, 6))

    ranking = pd.to_numeric(data.get("ranking_prioridad", pd.Series(dtype=int)), errors="coerce")
    risk_avg = pd.to_numeric(data.get("risk_avg_person", pd.Series(dtype=float)), errors="coerce")
    ranking_max = ranking.fillna(0).max() if not ranking.empty else 0
    risk_max = risk_avg.fillna(0.0).max() if not risk_avg.empty else 0.0
    if data.empty or (ranking_max <= 0 and risk_max <= 0.0):
        return _render_empty_chart(
            axis,
            "q18_user_risk_scores",
            timeframe,
            "Sin personas priorizadas por riesgo en el periodo seleccionado.",
        )

    work = data.copy()
    if "persona" not in work:
        work["persona"] = "sin_persona"
    work["persona"] = work["persona"].fillna("sin_persona").astype(str)

    if "risk_tier" not in work:
        work["risk_tier"] = "SIN_RIESGO"
    work["risk_tier"] = work["risk_tier"].fillna("SIN_RIESGO").astype(str)

    if "ranking_prioridad" in work:
        work["ranking_prioridad"] = (
            pd.to_numeric(work["ranking_prioridad"], errors="coerce").fillna(0).astype(int)
        )
    else:
        work["ranking_prioridad"] = list(range(1, len(work) + 1))

    if "risk_avg_person" in work:
        work["risk_avg_person"] = (
            pd.to_numeric(work["risk_avg_person"], errors="coerce").fillna(0.0)
        )
    else:
        work["risk_avg_person"] = 0.0

    if "net_flow" in work:
        work["net_flow"] = pd.to_numeric(work["net_flow"], errors="coerce").fillna(0.0)
    else:
        work["net_flow"] = 0.0

    if "movements" in work:
        work["movements"] = (
            pd.to_numeric(work["movements"], errors="coerce").fillna(0).astype(int)
        )
    else:
        work["movements"] = 0

    work = work.sort_values(
        ["ranking_prioridad", "risk_avg_person", "movements", "net_flow"],
        ascending=[True, False, False, False],
    ).head(max(1, int(top_n)))

    base_palette = {
        "ALTO": "#b22222",
        "MEDIO": "#ff8c00",
        "BAJO": "#1f77b4",
        "SIN_RIESGO": "#6c757d",
    }
    palette = {tier: base_palette.get(tier, "#17becf") for tier in work["risk_tier"].unique()}

    sns.barplot(
        data=work,
        x="risk_avg_person",
        y="persona",
        hue="risk_tier",
        palette=palette,
        dodge=False,
        ax=axis,
    )

    for patch, (_, row) in zip(axis.patches, work.iterrows()):
        axis.text(
            patch.get_width() + 0.02,
            patch.get_y() + patch.get_height() / 2,
            (
                f"rank {int(row.get('ranking_prioridad', 0))} · "
                f"mov={int(row.get('movements', 0))} · "
                f"net={row.get('net_flow', 0.0):,.0f}"
            ),
            va="center",
            fontsize=8,
            color="#333333",
        )

    _apply_plot_metadata(axis, "q18_user_risk_scores", timeframe)
    axis.set_xlabel("Riesgo promedio de la persona")
    axis.set_ylabel("Persona priorizada")
    _legend_outside(axis, title="Nivel de riesgo")
    return axis


__all__ = [
    "plot_q1_manager_nlp",
    "plot_q2_manager_concepts",
    "plot_q3_quid_pairs",
    "plot_q3_algo_pair_detalle",
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
    "plot_q18_user_risk_scores",
]
