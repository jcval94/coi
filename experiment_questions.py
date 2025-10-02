"""Módulo de experimentación para responder preguntas clave de COI."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping

try:  # pragma: no cover - dependencia opcional en tiempo de ejecución
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - guard para entornos sin deps
    raise SystemExit(
        "Este script requiere pandas. Ejecuta 'pip install -r requirements.txt' "
        "para instalar las dependencias necesarias."
    ) from exc

from coi_fraud import generate_diverse_dataset, run_pipeline
from coi_fraud.schemas import (
    COL_AMOUNT,
    COL_RECEIVER_ID,
    COL_RELATION,
    COL_SENDER_ID,
)


DEFAULT_TIMEFRAME = "todo_el_tiempo"
DEFAULT_OUTPUT_DIR = Path("answers")
NLP_CATEGORIES = ("SOBORNO", "FACILITACIÓN", "OFUSCACIÓN", "EXTORSIÓN", "FAVORES SEXUALES")
NLP_CATEGORY_SYNONYMS = {
    "SOBORNO": ("SOBOR", "COIMA", "BRIBE", "COHECHO"),
    "FACILITACIÓN": ("FACILIT", "FACILITATION", "GRATIFICACIÓN"),
    "OFUSCACIÓN": ("OFUSC", "OBFUS", "OCULT", "ENCUBR"),
    "EXTORSIÓN": ("EXTORS", "EXTORT", "AMENAZ"),
    "FAVORES SEXUALES": ("SEXUAL", "SEX", "ACOSO"),
}
CONCEPT_SPLIT_PATTERN = re.compile(r"[\s,;|/]+")


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _coalesce_str(*values: Any, default: str = "sin_valor") -> str:
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        text = str(value)
        if text and text.lower() != "nan":
            return text
    return default


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta la canalización completa y genera salidas enfocadas en las "
            "preguntas de experimentación prioritarias."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Ruta a un CSV de transacciones. Si no se proporciona se generará un "
            "dataset sintético diverso."
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10_000,
        help="Filas a generar cuando se use el dataset sintético (por defecto: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para el generador sintético (por defecto: 42).",
    )
    parser.add_argument(
        "--timeframe",
        choices=["ultimo_mes", "ultimos_3_meses", DEFAULT_TIMEFRAME],
        default=DEFAULT_TIMEFRAME,
        help="Ventana temporal sobre la cual responder las preguntas (por defecto: todo_el_tiempo).",
    )
    parser.add_argument(
        "--language",
        choices=["es", "en"],
        default="es",
        help="Idioma para la canalización NLP (por defecto: es).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directorio donde guardar los resultados en CSV (por defecto: ./answers). "
            "Cada pregunta generará un archivo con explicaciones detalladas."
        ),
    )
    return parser


def _load_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.input is not None:
        if not args.input.exists():
            raise SystemExit(f"El archivo de entrada '{args.input}' no existe.")
        return pd.read_csv(args.input)
    return generate_diverse_dataset(n_records=args.rows, seed=args.seed)


def _get_section(reports: Mapping[str, Any], section: str, timeframe: str) -> pd.DataFrame:
    data = reports.get(section, {})
    if isinstance(data, Mapping):
        df = data.get(timeframe)
    else:
        df = data
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return pd.DataFrame()


def _filter_manager_subordinate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or COL_RELATION not in df:
        return df.iloc[0:0].copy()
    mask = df[COL_RELATION].fillna("").astype(str).str.contains("manager", case=False)
    return df.loc[mask].copy()


def _tokenize_concepts(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    tokens = [
        re.sub(r"[^A-ZÁÉÍÓÚÜÑ0-9 ]", "", token.upper()).strip()
        for token in CONCEPT_SPLIT_PATTERN.split(text.upper())
    ]
    return [token for token in tokens if token]


def _coalesce_text_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df:
        return df[column].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def _manager_nlp_hits(
    tx: pd.DataFrame,
    categories: Iterable[str] = NLP_CATEGORIES,
) -> pd.DataFrame:
    work = _filter_manager_subordinate(tx)
    if work.empty:
        return work.iloc[0:0].copy()

    combined_text = (
        _coalesce_text_column(work, "nlp_concepto_sospechoso")
        + " "
        + _coalesce_text_column(work, "descripcion")
        + " "
        + _coalesce_text_column(work, "tx_tags")
    ).str.upper()

    hits: list[pd.DataFrame] = []
    for category in categories:
        synonyms = {category.upper()}
        synonyms.update(token.upper() for token in NLP_CATEGORY_SYNONYMS.get(category, ()))
        pattern = "|".join(re.escape(token) for token in sorted(synonyms, reverse=True) if token)
        if not pattern:
            continue
        mask = combined_text.str.contains(pattern, regex=True)
        if mask.any():
            subset = work.loc[mask].copy()
            subset["matched_category"] = category
            hits.append(subset)

    if hits:
        return pd.concat(hits, ignore_index=True, sort=False)

    if "nlp_tx_flag_concepto_sospechoso" in work:
        fallback = work.loc[work["nlp_tx_flag_concepto_sospechoso"].fillna(False).astype(bool)].copy()
        if not fallback.empty:
            fallback["matched_category"] = fallback["nlp_concepto_sospechoso"].apply(
                lambda value: next(iter(_tokenize_concepts(value)), "SOSPECHOSO")
            )
            return fallback

    return work.iloc[0:0].copy()


def question1_manager_nlp(
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
    categories: Iterable[str] = NLP_CATEGORIES,
) -> pd.DataFrame:
    tx = _get_section(reports, "transaccion", timeframe)
    if tx.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "month_id",
                "manager_user_id",
                "subordinado_user_id",
                "nlp_concepto_sospechoso",
                "tx_count",
                "monto_total",
                "interpretabilidad",
            ]
        )

    hits = _manager_nlp_hits(tx, categories)
    if hits.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "month_id",
                "manager_user_id",
                "subordinado_user_id",
                "nlp_concepto_sospechoso",
                "tx_count",
                "monto_total",
                "interpretabilidad",
            ]
        )

    hits["manager_user_id"] = (
        hits[COL_RECEIVER_ID].fillna("").astype(str).replace({"": pd.NA})
    )
    hits["subordinado_user_id"] = (
        hits[COL_SENDER_ID].fillna("").astype(str).replace({"": pd.NA})
    )
    hits = hits.dropna(subset=["manager_user_id", "subordinado_user_id"])
    if hits.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "month_id",
                "manager_user_id",
                "subordinado_user_id",
                "nlp_concepto_sospechoso",
                "tx_count",
                "monto_total",
                "interpretabilidad",
            ]
        )

    agg = (
        hits.groupby(
            [
                "month_id",
                "matched_category",
                "manager_user_id",
                "subordinado_user_id",
            ],
            observed=True,
        )
        .agg(
            tx_count=(COL_AMOUNT, "count"),
            monto_total=(COL_AMOUNT, "sum"),
        )
        .reset_index()
    )
    agg = agg.sort_values(["tx_count", "monto_total"], ascending=[False, False])
    agg["timeframe"] = timeframe
    agg = agg.rename(columns={"matched_category": "nlp_concepto_sospechoso"})
    agg["interpretabilidad"] = agg.apply(
        lambda row: (
            f"En la ventana '{timeframe}', durante {row.get('month_id', 'sin_mes')} "
            f"el manager {row.get('manager_user_id', 'sin_manager')} recibió "
            f"{int(row.get('tx_count', 0))} pagos del subordinado "
            f"{row.get('subordinado_user_id', 'sin_subordinado')} etiquetados como "
            f"'{row.get('nlp_concepto_sospechoso', 'SIN_CONCEPTO')}', acumulando "
            f"{_format_float(row.get('monto_total', 0))} en monto total."
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "month_id",
        "manager_user_id",
        "subordinado_user_id",
        "nlp_concepto_sospechoso",
        "tx_count",
        "monto_total",
        "interpretabilidad",
    ]
    return agg.reindex(columns=columns)


def question2_manager_concepts(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    tx = _get_section(reports, "transaccion", timeframe)
    if tx.empty or "risk_score" not in tx:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "month_id",
                "nlp_concepto_sospechoso",
                "tx_count",
                "risk_p95",
                "interpretabilidad",
            ]
        )

    hits = _manager_nlp_hits(tx, NLP_CATEGORIES)
    if hits.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "month_id",
                "nlp_concepto_sospechoso",
                "tx_count",
                "risk_p95",
                "interpretabilidad",
            ]
        )

    agg = (
        hits.groupby(["month_id", "matched_category"], observed=True)
        .agg(
            tx_count=("risk_score", "count"),
            risk_p95=("risk_score", lambda s: float(s.quantile(0.95)) if len(s) else 0.0),
        )
        .reset_index()
    )
    agg = agg.sort_values(["risk_p95", "tx_count"], ascending=[False, False])
    agg["timeframe"] = timeframe
    agg = agg.rename(columns={"matched_category": "nlp_concepto_sospechoso"})
    agg["interpretabilidad"] = agg.apply(
        lambda row: (
            f"En la ventana '{timeframe}', el concepto '{row.get('nlp_concepto_sospechoso', 'SIN_CONCEPTO')}' "
            f"tuvo {int(row.get('tx_count', 0))} transacciones manager-subordinado en {row.get('month_id', 'sin_mes')} "
            f"con severidad P95 de {row.get('risk_p95', 0.0):.2f}."
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "month_id",
        "nlp_concepto_sospechoso",
        "tx_count",
        "risk_p95",
        "interpretabilidad",
    ]
    return agg.reindex(columns=columns)


def question3_quid_pairs(
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
    min_score: float = 2.2,
    min_manager_ratio: float = 0.5,
) -> pd.DataFrame:
    pairs = _get_section(reports, "casuistica_quid_pro_quo_par", timeframe)
    tx = _get_section(reports, "casuistica_quid_pro_quo_tx", timeframe)
    base_tx = _get_section(reports, "transaccion", timeframe)

    pair_columns = [
        "quid_pair_clave",
        "quid_pair_label",
        "quid_tx_count",
        "quid_score_max",
        "quid_score_avg",
        "quid_manager_ratio",
        "quid_aprob_ratio",
        "quid_comp_ratio",
    ]

    if not pairs.empty:
        filtered_pairs = pairs.copy()
        filtered_pairs = filtered_pairs.loc[
            (filtered_pairs.get("quid_score_max", 0) >= min_score)
            & (filtered_pairs.get("quid_manager_ratio", 0) > min_manager_ratio)
            & (
                (filtered_pairs.get("quid_aprob_ratio", 0) > 0)
                | (filtered_pairs.get("quid_comp_ratio", 0) > 0)
            )
        ].copy()
        if not filtered_pairs.empty:
            filtered_pairs = filtered_pairs.sort_values(
                ["quid_score_max", "quid_tx_count"], ascending=[False, False]
            )
        pair_df = filtered_pairs.reindex(columns=pair_columns)
    else:
        pair_df = pd.DataFrame(columns=pair_columns)

    relaxed_pairs = pd.DataFrame()
    if pair_df.empty and not base_tx.empty:
        fallback_candidates = base_tx.loc[
            (base_tx.get("feat_quid_score", 0).fillna(0) >= min_score)
            | base_tx.get("sig_quid_pro_quo", False).fillna(False)
        ].copy()
        if fallback_candidates.empty:
            fallback_candidates = base_tx.copy()
        if not fallback_candidates.empty:
            fallback_candidates["es_manager"] = fallback_candidates.get(COL_RELATION, "").astype(str).str.contains(
                "manager", case=False
            )
            if "feat_quid_has_approval" not in fallback_candidates:
                fallback_candidates["feat_quid_has_approval"] = False
            if "feat_quid_has_comp" not in fallback_candidates:
                fallback_candidates["feat_quid_has_comp"] = False
            fallback_pairs = (
                fallback_candidates.groupby([COL_SENDER_ID, COL_RECEIVER_ID], observed=True)
                .agg(
                    quid_tx_count=(COL_AMOUNT, "count"),
                    quid_score_max=("feat_quid_score", "max"),
                    quid_score_avg=("feat_quid_score", "mean"),
                    quid_manager_ratio=("es_manager", "mean"),
                    quid_aprob_ratio=("feat_quid_has_approval", "mean"),
                    quid_comp_ratio=("feat_quid_has_comp", "mean"),
                )
                .reset_index()
            )
            fallback_pairs["quid_pair_clave"] = (
                fallback_pairs[COL_SENDER_ID].astype(str) + "->" + fallback_pairs[COL_RECEIVER_ID].astype(str)
            )
            fallback_pairs["quid_pair_label"] = fallback_pairs["quid_pair_clave"]
            fallback_pairs = fallback_pairs.loc[
                (fallback_pairs["quid_score_max"].fillna(0) >= min_score)
                & (fallback_pairs["quid_manager_ratio"].fillna(0) > min_manager_ratio)
            ].copy()
            if not fallback_pairs.empty:
                fallback_pairs["criterio_relajado"] = False
                pair_df = fallback_pairs.reindex(columns=pair_columns + ["criterio_relajado"])
            else:
                relaxed_pairs = (
                    fallback_candidates.groupby([COL_SENDER_ID, COL_RECEIVER_ID], observed=True)
                    .agg(
                        quid_tx_count=(COL_AMOUNT, "count"),
                        quid_score_max=("feat_quid_score", "max"),
                        quid_score_avg=("feat_quid_score", "mean"),
                        quid_manager_ratio=("es_manager", "mean"),
                        quid_aprob_ratio=("feat_quid_has_approval", "mean"),
                        quid_comp_ratio=("feat_quid_has_comp", "mean"),
                    )
                    .reset_index()
                )
                relaxed_pairs["quid_pair_clave"] = (
                    relaxed_pairs[COL_SENDER_ID].astype(str)
                    + "->"
                    + relaxed_pairs[COL_RECEIVER_ID].astype(str)
                )
                relaxed_pairs["quid_pair_label"] = relaxed_pairs["quid_pair_clave"]
                relaxed_pairs["criterio_relajado"] = True
                relaxed_pairs = relaxed_pairs.sort_values(
                    ["quid_score_max", "quid_tx_count"], ascending=[False, False]
                ).head(10)
                pair_df = relaxed_pairs.reindex(columns=pair_columns + ["criterio_relajado"])

    pair_df["nivel_respuesta"] = "par"
    pair_df["timeframe"] = timeframe
    if not pair_df.empty:
        pair_df["interpretabilidad"] = pair_df.apply(
            lambda row: (
                f"El par '{_coalesce_str(row.get('quid_pair_label'), row.get('quid_pair_clave'), default='sin_identificar')}' "
                f"acumuló {int(row.get('quid_tx_count', 0))} transacciones con puntaje máximo "
                f"{row.get('quid_score_max', 0):.2f} (≥{min_score}), promedio {row.get('quid_score_avg', 0):.2f} y "
                f"ratio jerárquico {row.get('quid_manager_ratio', 0):.2f}; aprobaciones dentro de la ventana: "
                f"{row.get('quid_aprob_ratio', 0):.2f}, compensaciones {row.get('quid_comp_ratio', 0):.2f}."
                + (
                    " Se relajó el umbral para destacar los puntajes más altos disponibles."  # type: ignore[arg-type]
                    if row.get("criterio_relajado")
                    else ""
                )
            ),
            axis=1,
        )
    else:
        pair_df["interpretabilidad"] = pd.Series(dtype="object")
    if "criterio_relajado" in pair_df:
        pair_df = pair_df.drop(columns=["criterio_relajado"])

    tx_columns = [
        "fecha_hora_ts",
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        COL_AMOUNT,
        "relacion",
        "feat_quid_rel_label",
        "feat_quid_has_approval",
        "feat_quid_has_comp",
        "feat_quid_value_vs_load_days",
        "feat_quid_score",
        "descripcion",
        "reference_number_trans_desc",
        "risk_score",
    ]

    if not tx.empty:
        filtered_tx = tx.loc[tx.get("feat_quid_score", 0) >= min_score].copy()
    elif not base_tx.empty:
        filtered_tx = base_tx.loc[base_tx.get("feat_quid_score", 0) >= min_score].copy()
    else:
        filtered_tx = pd.DataFrame(columns=tx_columns)

    if not filtered_tx.empty:
        filtered_tx = filtered_tx.sort_values(
            ["feat_quid_score", "fecha_hora_ts"], ascending=[False, True]
        )
        tx_df = filtered_tx.reindex(columns=tx_columns)
        tx_df["criterio_relajado"] = False
    elif not base_tx.empty:
        relaxed_tx = base_tx.copy()
        relaxed_tx = relaxed_tx.sort_values(
            ["feat_quid_score", "fecha_hora_ts"], ascending=[False, True]
        ).head(10)
        tx_df = relaxed_tx.reindex(columns=tx_columns)
        tx_df["criterio_relajado"] = True
    else:
        tx_df = pd.DataFrame(columns=tx_columns)

    tx_df["nivel_respuesta"] = "transaccion"
    tx_df["timeframe"] = timeframe
    if not tx_df.empty:
        tx_df["interpretabilidad"] = tx_df.apply(
            lambda row: (
                f"La transacción del {row.get('fecha_hora_ts', 'sin_fecha')} entre "
                f"{_coalesce_str(row.get(COL_SENDER_ID), default='emisor_desconocido')} y "
                f"{_coalesce_str(row.get(COL_RECEIVER_ID), default='receptor_desconocido')} "
                f"alcanzó un puntaje quid-pro-quo de {row.get('feat_quid_score', 0):.2f} (umbral {min_score}), "
                f"con valor {_format_float(row.get(COL_AMOUNT, 0))} y desfase autorización-carga "
                f"de {row.get('feat_quid_value_vs_load_days', 0)} días; aprobaciones asociadas: "
                f"{bool(row.get('feat_quid_has_approval', False))}, compensaciones: "
                f"{bool(row.get('feat_quid_has_comp', False))}."
                + (
                    " Se listó con umbral relajado al no hallarse casos ≥ objetivo."  # type: ignore[arg-type]
                    if row.get("criterio_relajado")
                    else ""
                )
            ),
            axis=1,
        )
    else:
        tx_df["interpretabilidad"] = pd.Series(dtype="object")
    if "criterio_relajado" in tx_df:
        tx_df = tx_df.drop(columns=["criterio_relajado"])

    combined = pd.concat([pair_df, tx_df], ignore_index=True, sort=False)
    ordered_cols = ["timeframe", "nivel_respuesta"] + [c for c in pair_columns + tx_columns if c in combined.columns]
    ordered_cols = list(dict.fromkeys(ordered_cols)) + ["interpretabilidad"]
    return combined.reindex(columns=ordered_cols)


def question4_quid_negative_value_vs_load(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    tx = _get_section(reports, "casuistica_quid_pro_quo_tx", timeframe)
    base_tx = _get_section(reports, "transaccion", timeframe)
    if tx.empty and not base_tx.empty:
        tx = base_tx.copy()
    if tx.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "month_id",
                "fecha_hora_ts",
                COL_SENDER_ID,
                COL_RECEIVER_ID,
                COL_RELATION,
                "feat_quid_rel_label",
                "feat_quid_score",
                "feat_quid_value_vs_load_days",
                "feat_quid_has_approval",
                "feat_quid_has_comp",
                "responsable_user_id",
                "interpretabilidad",
            ]
        )

    if "feat_quid_value_vs_load_days" not in tx:
        tx = tx.copy()
        tx["feat_quid_value_vs_load_days"] = pd.NA

    filtered = tx.loc[tx["feat_quid_value_vs_load_days"] < 0].copy()
    relaxed = False
    if filtered.empty:
        relaxed = True
        fallback_source = tx.loc[tx["feat_quid_value_vs_load_days"].notna()].copy()
        if fallback_source.empty:
            filtered = tx.sort_values("feat_quid_score", ascending=False).head(10)
        else:
            filtered = fallback_source.sort_values(
                "feat_quid_value_vs_load_days", ascending=True
            ).head(10)

    base_cols = [
        "fecha_hora_ts",
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        COL_RELATION,
        "feat_quid_rel_label",
        "feat_quid_score",
        "feat_quid_value_vs_load_days",
        "feat_quid_has_approval",
        "feat_quid_has_comp",
    ]
    if "month_id" in filtered:
        cols = ["month_id"] + base_cols
    else:
        cols = base_cols
    keep = [c for c in cols if c in filtered.columns]
    result = filtered[keep].sort_values(
        ["feat_quid_value_vs_load_days", "feat_quid_score"], ascending=[True, False]
    )
    result["timeframe"] = timeframe

    def _responsable(row: pd.Series) -> str:
        relation = str(row.get(COL_RELATION, "")).lower()
        rel_label = str(row.get("feat_quid_rel_label", "")).lower()
        sender = row.get(COL_SENDER_ID)
        receiver = row.get(COL_RECEIVER_ID)
        if "manager" in relation or "manager" in rel_label:
            if relation == "manager_del_emisor" or "->manager" in rel_label:
                candidate = receiver if receiver else sender
            else:
                candidate = sender if sender else receiver
        else:
            candidate = receiver if receiver else sender
        return _coalesce_str(candidate, default="sin_responsable")

    result["responsable_user_id"] = result.apply(_responsable, axis=1)
    def _describe_quid(row: pd.Series) -> str:
        delta = row.get("feat_quid_value_vs_load_days")
        if pd.notna(delta) and float(delta) < 0:
            delta_txt = f"mostró autorizaciones previas con {int(delta)} días negativos"
        elif pd.notna(delta):
            delta_txt = f"presentó un desfase de {int(delta)} días (no negativo)"
        else:
            delta_txt = "no cuenta con desfase registrado"
        base = (
            f"En la ventana '{timeframe}', la transacción del {row.get('fecha_hora_ts', 'sin_fecha')} "
            f"entre {_coalesce_str(row.get(COL_SENDER_ID), default='emisor_desconocido')} y "
            f"{_coalesce_str(row.get(COL_RECEIVER_ID), default='receptor_desconocido')} {delta_txt}, "
            f"con puntaje quid-pro-quo {row.get('feat_quid_score', 0):.2f}. "
            f"Se señala como responsable a {row.get('responsable_user_id', 'sin_responsable')}."
        )
        if relaxed:
            base += " Se listan los casos con los menores desfases/puntajes disponibles para seguimiento preventivo."
        return base

    result["interpretabilidad"] = result.apply(_describe_quid, axis=1)
    ordered_cols = ["timeframe"] + keep + ["responsable_user_id", "interpretabilidad"]
    return result.reindex(columns=ordered_cols)


def question5_reference_reuse(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    summary = _get_section(reports, "casuistica_referencia_resumen", timeframe)
    tx = _get_section(reports, "casuistica_referencia_tx", timeframe)
    base_tx = _get_section(reports, "transaccion", timeframe)

    summary_columns = [
        "reference_norm",
        "reference_len",
        "first_ts",
        "last_ts",
        "days_range",
        "n_pairs",
        "pairs",
        "tx_count",
    ]

    summary_relajado = False
    if not summary.empty:
        filtered = summary.loc[summary.get("n_pairs", 0) > 1].copy()
        filtered = filtered.sort_values(
            ["n_pairs", "days_range", "tx_count"], ascending=[False, True, False]
        )
        summary_df = filtered.reindex(columns=summary_columns)
    else:
        summary_df = pd.DataFrame(columns=summary_columns)
        if not base_tx.empty and "feat_reference_norm" in base_tx:
            base_refs = base_tx.copy()
            base_refs["reference_norm"] = base_refs.get("feat_reference_norm", "").fillna("").astype(str)
            if base_refs["reference_norm"].str.strip().eq("").all():
                base_refs["reference_norm"] = (
                    base_refs.get("descripcion", "")
                    .fillna("")
                    .astype(str)
                    .str.lower()
                    .str.replace(r"[^a-z0-9]+", " ", regex=True)
                    .str.strip()
                )
            base_refs = base_refs.loc[base_refs["reference_norm"].str.len() > 0].copy()
            if not base_refs.empty:
                if "feat_reference_len" not in base_refs:
                    base_refs["feat_reference_len"] = base_refs["reference_norm"].str.len()
                base_refs["fecha_hora_ts"] = pd.to_datetime(
                    base_refs.get("fecha_hora_ts"), errors="coerce"
                )
                base_refs["pair_id"] = (
                    base_refs.get(COL_SENDER_ID, "").astype(str)
                    + "->"
                    + base_refs.get(COL_RECEIVER_ID, "").astype(str)
                )
                fallback_summary = (
                    base_refs.groupby("reference_norm", observed=True)
                    .agg(
                        reference_len=("feat_reference_len", "max"),
                        first_ts=("fecha_hora_ts", "min"),
                        last_ts=("fecha_hora_ts", "max"),
                        n_pairs=("pair_id", "nunique"),
                        pairs=("pair_id", lambda s: "; ".join(sorted(set(map(str, s))))),
                        tx_count=(COL_AMOUNT, "count"),
                    )
                    .reset_index()
                )
                if not fallback_summary.empty:
                    fallback_summary["days_range"] = (
                        fallback_summary["last_ts"] - fallback_summary["first_ts"]
                    ).dt.days.fillna(0)
                    fallback_summary = fallback_summary.loc[
                        fallback_summary["n_pairs"] > 1
                    ].copy()
                    fallback_summary = fallback_summary.loc[
                        fallback_summary["days_range"].fillna(0) <= 30
                    ]
                    fallback_summary["first_ts"] = fallback_summary["first_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
                    fallback_summary["last_ts"] = fallback_summary["last_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
                    summary_df = fallback_summary.reindex(columns=summary_columns)
                    summary_relajado = False
                if summary_df.empty:
                    relaxed_summary = (
                        base_refs.groupby("reference_norm", observed=True)
                        .agg(
                            reference_len=("feat_reference_len", "max"),
                            first_ts=("fecha_hora_ts", "min"),
                            last_ts=("fecha_hora_ts", "max"),
                            n_pairs=("pair_id", "nunique"),
                            pairs=("pair_id", lambda s: "; ".join(sorted(set(map(str, s))))),
                            tx_count=(COL_AMOUNT, "count"),
                        )
                        .reset_index()
                    )
                    if not relaxed_summary.empty:
                        relaxed_summary["days_range"] = (
                            relaxed_summary["last_ts"] - relaxed_summary["first_ts"]
                        ).dt.days.fillna(0)
                        relaxed_summary = relaxed_summary.loc[relaxed_summary["tx_count"] > 1].copy()
                        relaxed_summary["first_ts"] = relaxed_summary["first_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
                        relaxed_summary["last_ts"] = relaxed_summary["last_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
                        relaxed_summary = relaxed_summary.sort_values(
                            ["tx_count", "days_range"], ascending=[False, True]
                        ).head(10)
                        relaxed_summary["n_pairs"] = relaxed_summary["n_pairs"].fillna(1)
                        summary_df = relaxed_summary.reindex(columns=summary_columns)
                        summary_relajado = True

    summary_df["nivel_respuesta"] = "referencia"
    summary_df["timeframe"] = timeframe
    if not summary_df.empty:
        summary_df["interpretabilidad"] = summary_df.apply(
            lambda row: (
                f"La referencia normalizada '{_coalesce_str(row.get('reference_norm'), default='sin_referencia')}' se reutilizó en "
                f"{int(row.get('n_pairs', 0))} pares dentro de {row.get('days_range', 0)} días, "
                f"generando {int(row.get('tx_count', 0))} transacciones entre {row.get('first_ts', 'sin_fecha')} "
                f"y {row.get('last_ts', 'sin_fecha')}."
                + (
                    " Se listan referencias recurrentes sin cumplir aún el criterio multi-par (modo relajado)."
                    if summary_relajado
                    else ""
                )
            ),
            axis=1,
        )
    else:
        summary_df["interpretabilidad"] = pd.Series(dtype="object")

    tx_columns = [
        "fecha_hora_ts",
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        "reference_number_trans_desc",
        "feat_reference_norm",
        "feat_reference_len",
        "risk_score",
    ]

    if not tx.empty:
        involved_refs = summary_df["reference_norm"].dropna().unique().tolist()
        filtered_tx = tx.loc[tx.get("feat_reference_norm", "").isin(involved_refs)].copy()
    elif not base_tx.empty and "feat_reference_norm" in base_tx:
        involved_refs = summary_df["reference_norm"].dropna().unique().tolist()
        filtered_tx = base_tx.loc[base_tx.get("feat_reference_norm", "").isin(involved_refs)].copy()
    else:
        tx_df = pd.DataFrame(columns=tx_columns)

    if 'filtered_tx' in locals() and not filtered_tx.empty:
        filtered_tx = filtered_tx.sort_values(
            ["feat_reference_norm", "fecha_hora_ts"], ascending=[True, True]
        )
        tx_df = filtered_tx.reindex(columns=tx_columns)
    else:
        tx_df = pd.DataFrame(columns=tx_columns)

    tx_df["nivel_respuesta"] = "transaccion"
    tx_df["timeframe"] = timeframe
    if not tx_df.empty:
        tx_df["interpretabilidad"] = tx_df.apply(
            lambda row: (
                f"La transacción del {row.get('fecha_hora_ts', 'sin_fecha')} reutilizó la referencia "
                f"'{_coalesce_str(row.get('feat_reference_norm'), default='sin_referencia')}' entre "
                f"{_coalesce_str(row.get(COL_SENDER_ID), default='emisor_desconocido')} y "
                f"{_coalesce_str(row.get(COL_RECEIVER_ID), default='receptor_desconocido')}, "
                f"sugiriendo coordinación/ofuscación en la ventana '{timeframe}'."
            ),
            axis=1,
        )
    else:
        tx_df["interpretabilidad"] = pd.Series(dtype="object")

    combined = pd.concat([summary_df, tx_df], ignore_index=True, sort=False)
    ordered_cols = [
        "timeframe",
        "nivel_respuesta",
    ] + [c for c in summary_columns + tx_columns if c in combined.columns]
    ordered_cols = list(dict.fromkeys(ordered_cols)) + ["interpretabilidad"]
    return combined.reindex(columns=ordered_cols)


def question6_centralizers(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    tx = _get_section(reports, "transaccion", timeframe)
    if tx.empty or "month_id" not in tx:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "month_id",
                COL_RECEIVER_ID,
                "inflow",
                "emisores_unicos",
                "n_tx",
                "risk_avg",
                "centralidad",
                "interpretabilidad",
            ]
        )

    work = tx[["month_id", COL_SENDER_ID, COL_RECEIVER_ID, COL_AMOUNT, "risk_score"]].copy()
    agg = (
        work.groupby(["month_id", COL_RECEIVER_ID], observed=True)
        .agg(
            inflow=(COL_AMOUNT, "sum"),
            emisores_unicos=(COL_SENDER_ID, "nunique"),
            n_tx=(COL_AMOUNT, "count"),
            risk_avg=("risk_score", "mean"),
        )
        .reset_index()
    )
    agg["centralidad"] = agg["inflow"] * agg["emisores_unicos"]
    agg = agg.sort_values(["month_id", "centralidad"], ascending=[True, False])
    agg["timeframe"] = timeframe
    agg["interpretabilidad"] = agg.apply(
        lambda row: (
            f"En {_coalesce_str(row.get('month_id'), default='sin_mes')} ({timeframe}), el receptor "
            f"{_coalesce_str(row.get(COL_RECEIVER_ID), default='sin_receptor')} "
            f"recibió {_format_float(row.get('inflow', 0))} de {int(row.get('emisores_unicos', 0))} emisores únicos "
            f"a través de {int(row.get('n_tx', 0))} pagos, logrando centralidad {row.get('centralidad', 0):.2f} "
            f"y riesgo promedio {row.get('risk_avg', 0):.2f}."
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "month_id",
        COL_RECEIVER_ID,
        "inflow",
        "emisores_unicos",
        "n_tx",
        "risk_avg",
        "centralidad",
        "interpretabilidad",
    ]
    return agg.reindex(columns=columns)


def question7_net_imbalance(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    personas = _get_section(reports, "persona", timeframe)
    if personas.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "persona",
                "desbalance_persona_monto_neto",
                "desbalance_persona_meses_envia_extremo",
                "desbalance_persona_meses_recibe_extremo",
                "interpretabilidad",
            ]
        )

    cols = [
        "persona",
        "desbalance_persona_monto_neto",
        "desbalance_persona_meses_envia_extremo",
        "desbalance_persona_meses_recibe_extremo",
    ]
    keep = [c for c in cols if c in personas.columns]
    work = personas[keep].copy()
    if "desbalance_persona_monto_neto" not in work:
        work["desbalance_persona_monto_neto"] = 0.0
    work["abs_neto"] = work["desbalance_persona_monto_neto"].abs()
    work = work.sort_values("abs_neto", ascending=False)
    work["timeframe"] = timeframe
    work["interpretabilidad"] = work.apply(
        lambda row: (
            f"En la ventana '{timeframe}', la persona {_coalesce_str(row.get('persona'), default='sin_persona')} "
            f"presenta un desbalance neto de {_format_float(row.get('desbalance_persona_monto_neto', 0))}, "
            f"con meses extremos al enviar: {int(row.get('desbalance_persona_meses_envia_extremo', 0))} "
            f"y al recibir: {int(row.get('desbalance_persona_meses_recibe_extremo', 0))}."
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "persona",
        "desbalance_persona_monto_neto",
        "desbalance_persona_meses_envia_extremo",
        "desbalance_persona_meses_recibe_extremo",
        "interpretabilidad",
    ]
    return work.reindex(columns=columns)


def _run_questions(reports: Mapping[str, Any], timeframe: str) -> Dict[str, Any]:
    return {
        "q1_manager_nlp": question1_manager_nlp(reports, timeframe),
        "q2_manager_concepts": question2_manager_concepts(reports, timeframe),
        "q3_quid_pairs": question3_quid_pairs(reports, timeframe),
        "q4_quid_negative_value_vs_load": question4_quid_negative_value_vs_load(reports, timeframe),
        "q5_reference_reuse": question5_reference_reuse(reports, timeframe),
        "q6_centralizers": question6_centralizers(reports, timeframe),
        "q7_net_imbalance": question7_net_imbalance(reports, timeframe),
    }


def _export_results(results: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"{key}.csv", index=False)
        else:
            raise TypeError(
                f"El resultado de {key} no es un DataFrame y no puede exportarse."
            )


def _print_summary(results: Mapping[str, Any]) -> None:
    for key, value in results.items():
        print(f"\n== {key} ==")
        if isinstance(value, pd.DataFrame):
            print(value.head())
        else:
            print(value)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    df = _load_dataframe(args)
    reports = run_pipeline(df, language=args.language)
    results = _run_questions(reports, args.timeframe)

    if args.output_dir is not None:
        _export_results(results, args.output_dir)

    _print_summary(results)


if __name__ == "__main__":
    main()
