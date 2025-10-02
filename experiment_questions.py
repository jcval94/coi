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


def question8_case13_new_employees(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    personas = _get_section(reports, "persona", timeframe)
    required = [
        "persona",
        "caso13_persona_tx_recibidas",
        "caso13_persona_monto_total",
        "caso13_persona_emisores_unicos",
        "caso13_persona_tx_altos",
        "caso13_persona_monto_promedio",
        "caso13_persona_flag_nuevo_receptor_altos_montos",
    ]
    if personas.empty or not set(required).issubset(personas.columns):
        return pd.DataFrame(
            columns=[
                "timeframe",
                "persona",
                "caso13_persona_tx_recibidas",
                "caso13_persona_monto_total",
                "caso13_persona_emisores_unicos",
                "caso13_persona_tx_altos",
                "caso13_persona_monto_promedio",
                "interpretabilidad",
            ]
        )

    work = personas[required].copy()
    mask = work["caso13_persona_flag_nuevo_receptor_altos_montos"].fillna(0).astype(int) > 0
    work = work.loc[mask].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "persona",
                "caso13_persona_tx_recibidas",
                "caso13_persona_monto_total",
                "caso13_persona_emisores_unicos",
                "caso13_persona_tx_altos",
                "caso13_persona_monto_promedio",
                "interpretabilidad",
            ]
        )

    work["timeframe"] = timeframe
    work = work.sort_values(
        ["caso13_persona_tx_altos", "caso13_persona_monto_total"], ascending=[False, False]
    )
    work["interpretabilidad"] = work.apply(
        lambda row: (
            f"En la ventana '{timeframe}', la persona {_coalesce_str(row.get('persona'), default='sin_persona')} "
            f"es un receptor con antigüedad ≤6 meses que recibió "
            f"{int(row.get('caso13_persona_tx_recibidas', 0))} transferencias totales "
            f"desde {int(row.get('caso13_persona_emisores_unicos', 0))} emisores únicos, "
            f"de las cuales {int(row.get('caso13_persona_tx_altos', 0))} fueron de monto alto (percentil 90). "
            f"El flujo acumulado asciende a {_format_float(row.get('caso13_persona_monto_total', 0))} "
            f"con un promedio por transacción de {_format_float(row.get('caso13_persona_monto_promedio', 0))}."
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "persona",
        "caso13_persona_tx_recibidas",
        "caso13_persona_monto_total",
        "caso13_persona_emisores_unicos",
        "caso13_persona_tx_altos",
        "caso13_persona_monto_promedio",
        "interpretabilidad",
    ]
    return work.reindex(columns=columns)


def question9_case14_veterans_from_newcomers(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    personas = _get_section(reports, "persona", timeframe)
    required = [
        "persona",
        "caso14_persona_tx_de_emisores_nuevos",
        "caso14_persona_monto_de_emisores_nuevos",
        "caso14_persona_emisores_nuevos_unicos",
        "caso14_persona_monto_promedio_de_emisores_nuevos",
        "caso14_persona_flag_antiguo_recibe_de_nuevos",
    ]
    if personas.empty or not set(required).issubset(personas.columns):
        return pd.DataFrame(
            columns=[
                "timeframe",
                "persona",
                "caso14_persona_tx_de_emisores_nuevos",
                "caso14_persona_monto_de_emisores_nuevos",
                "caso14_persona_emisores_nuevos_unicos",
                "caso14_persona_monto_promedio_de_emisores_nuevos",
                "interpretabilidad",
            ]
        )

    work = personas[required].copy()
    mask = work["caso14_persona_flag_antiguo_recibe_de_nuevos"].fillna(0).astype(int) > 0
    work = work.loc[mask].copy()
    relaxed = False
    heuristic = False

    if work.empty:
        relaxed = True
        fallback_mask = (
            work_original := personas[required].copy()
        )["caso14_persona_tx_de_emisores_nuevos"].fillna(0).astype(int) > 0
        fallback_mask &= (
            work_original["caso14_persona_emisores_nuevos_unicos"].fillna(0).astype(int)
            > 0
        )
        work = work_original.loc[fallback_mask].copy()

    if work.empty:
        tx = _get_section(reports, "transaccion", timeframe)

        def _heuristic_case14(transacciones: pd.DataFrame) -> pd.DataFrame:
            if transacciones.empty:
                return pd.DataFrame()
            cols = {
                COL_SENDER_ID,
                COL_RECEIVER_ID,
                COL_AMOUNT,
                "user_antiguedad_anios",
                "receptor_antiguedad_anios",
            }
            if not cols.issubset(transacciones.columns):
                return pd.DataFrame()
            base = transacciones[list(cols)].copy()
            base["user_antiguedad_anios"] = pd.to_numeric(
                base["user_antiguedad_anios"], errors="coerce"
            )
            base["receptor_antiguedad_anios"] = pd.to_numeric(
                base["receptor_antiguedad_anios"], errors="coerce"
            )
            base = base.dropna(subset=["user_antiguedad_anios", "receptor_antiguedad_anios"])
            if base.empty:
                return pd.DataFrame()

            thresholds = [
                (0.5, 5.0),
                (1.0, 4.0),
                (1.5, 3.5),
            ]
            selected: pd.DataFrame | None = None
            applied: tuple[float, float] | None = None
            for young, veteran in thresholds:
                mask_new = base["user_antiguedad_anios"] <= young
                mask_old = base["receptor_antiguedad_anios"] >= veteran
                candidate = base.loc[mask_new & mask_old].copy()
                if not candidate.empty:
                    selected = candidate
                    applied = (young, veteran)
                    break
            if selected is None:
                quantiles = base[["user_antiguedad_anios", "receptor_antiguedad_anios"]].quantile(
                    [0.25, 0.75]
                )
                young_q = float(quantiles.loc[0.25, "user_antiguedad_anios"])
                veteran_q = float(quantiles.loc[0.75, "receptor_antiguedad_anios"])
                candidate = base.loc[
                    (base["user_antiguedad_anios"] <= young_q)
                    & (base["receptor_antiguedad_anios"] >= veteran_q)
                ].copy()
                if candidate.empty():
                    candidate = base.nsmallest(100, "user_antiguedad_anios").copy()
                selected = candidate
                applied = (young_q, veteran_q)
            if selected is None or selected.empty():
                return pd.DataFrame()

            grouped = (
                selected.groupby(COL_RECEIVER_ID, observed=True)
                .agg(
                    caso14_persona_tx_de_emisores_nuevos=(COL_AMOUNT, "count"),
                    caso14_persona_monto_de_emisores_nuevos=(COL_AMOUNT, "sum"),
                    caso14_persona_emisores_nuevos_unicos=(COL_SENDER_ID, "nunique"),
                )
                .reset_index()
            )
            grouped.rename(columns={COL_RECEIVER_ID: "persona"}, inplace=True)
            grouped["caso14_persona_monto_promedio_de_emisores_nuevos"] = (
                grouped["caso14_persona_monto_de_emisores_nuevos"]
                / grouped["caso14_persona_tx_de_emisores_nuevos"].replace(0, pd.NA)
            )
            grouped["caso14_persona_monto_promedio_de_emisores_nuevos"] = (
                grouped["caso14_persona_monto_promedio_de_emisores_nuevos"].fillna(0.0)
            )
            if applied is not None:
                grouped["_heuristic_note"] = (
                    f"emisor≤{applied[0]:.1f}a / receptor≥{applied[1]:.1f}a"
                )
            return grouped

        heuristic_df = _heuristic_case14(tx)
        if not heuristic_df.empty:
            heuristic = True
            if "_heuristic_note" in heuristic_df:
                heuristic_df["_heuristic_note"] = heuristic_df["_heuristic_note"].astype(str)
            else:
                heuristic_df["_heuristic_note"] = ""
            work = heuristic_df

    if work.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "persona",
                "caso14_persona_tx_de_emisores_nuevos",
                "caso14_persona_monto_de_emisores_nuevos",
                "caso14_persona_emisores_nuevos_unicos",
                "caso14_persona_monto_promedio_de_emisores_nuevos",
                "interpretabilidad",
            ]
        )

    work["timeframe"] = timeframe
    work = work.sort_values(
        ["caso14_persona_tx_de_emisores_nuevos", "caso14_persona_monto_de_emisores_nuevos"],
        ascending=[False, False],
    )
    work["interpretabilidad"] = work.apply(
        lambda row: (
            f"Dentro de '{timeframe}', la persona {_coalesce_str(row.get('persona'), default='sin_persona')} "
            f"(antigüedad ≥5 años) recibió {int(row.get('caso14_persona_tx_de_emisores_nuevos', 0))} pagos "
            f"desde recién ingresados (≤6 meses), provenientes de "
            f"{int(row.get('caso14_persona_emisores_nuevos_unicos', 0))} emisores distintos. "
            f"Estos movimientos suman {_format_float(row.get('caso14_persona_monto_de_emisores_nuevos', 0))} "
            f"con un promedio individual de {_format_float(row.get('caso14_persona_monto_promedio_de_emisores_nuevos', 0))}."
            + (
                " Se relajó el criterio original para mostrar relaciones con indicios "
                "incipientes." if relaxed and not row.get("caso14_persona_flag_antiguo_recibe_de_nuevos") else ""
            )
            + (
                (" Se aplicó una heurística por antigüedad relativa." if heuristic else "")
                if not heuristic
                else (
                    f" Heurística por antigüedad relativa ({row.get('_heuristic_note')}) activada."
                    if row.get("_heuristic_note")
                    else " Se aplicó una heurística por antigüedad relativa."
                )
            )
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "persona",
        "caso14_persona_tx_de_emisores_nuevos",
        "caso14_persona_monto_de_emisores_nuevos",
        "caso14_persona_emisores_nuevos_unicos",
        "caso14_persona_monto_promedio_de_emisores_nuevos",
        "interpretabilidad",
    ]
    return work.reindex(columns=columns)


def question10_yoyo_streaks(
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
    min_consecutive: int = 2,
    risk_threshold: float = 1.8,
) -> pd.DataFrame:
    tx = _get_section(reports, "transaccion", timeframe)
    required = [
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        "fecha_hora_ts",
        "month_id",
        "sig_yoyo",
        "risk_score",
    ]
    if tx.empty or not set(required).issubset(tx.columns):
        return pd.DataFrame(
            columns=[
                "timeframe",
                "par_bidir",
                "racha_max_yo_yo",
                "tx_yo_yo_totales",
                "meses_con_yo_yo",
                "riesgo_max_par",
                "riesgo_promedio_par",
                "riesgo_max_yo_yo",
                "riesgo_promedio_yo_yo",
                "interpretabilidad",
            ]
        )

    work = tx[required].copy()
    work["sig_yoyo"] = work["sig_yoyo"].fillna(False).astype(bool)
    heuristic = False
    heuristic_note = None
    if not work["sig_yoyo"].any():
        heuristic = True
        work["par_bidir"] = work.apply(
            lambda row: "⇄".join(
                sorted([str(row[COL_SENDER_ID]), str(row[COL_RECEIVER_ID])])
            ),
            axis=1,
        )

        def _derive_flags(group: pd.DataFrame, hour_limit: float) -> pd.DataFrame:
            ordered = group.sort_values("fecha_hora_ts").copy()
            flags = [False] * len(ordered)
            timestamps = ordered["fecha_hora_ts"].tolist()
            senders = ordered[COL_SENDER_ID].astype(str).tolist()
            receivers = ordered[COL_RECEIVER_ID].astype(str).tolist()
            for idx in range(len(ordered)):
                curr_ts = timestamps[idx]
                curr_sender = senders[idx]
                curr_receiver = receivers[idx]
                for prev_idx in range(idx - 1, -1, -1):
                    delta_hours = (curr_ts - timestamps[prev_idx]).total_seconds() / 3600.0
                    if delta_hours > hour_limit:
                        break
                    if (
                        senders[prev_idx] == curr_receiver
                        and receivers[prev_idx] == curr_sender
                    ):
                        flags[idx] = True
                        flags[prev_idx] = True
                        break
            ordered["sig_yoyo"] = flags
            return ordered

        candidate = None
        for hour_limit in (8.0, 24.0, 72.0):
            derived = (
                work.groupby("par_bidir", observed=True, group_keys=False)
                .apply(lambda g: _derive_flags(g, hour_limit))
                .reset_index(drop=True)
            )
            derived["sig_yoyo"] = derived["sig_yoyo"].fillna(False).astype(bool)
            if derived["sig_yoyo"].any():
                candidate = derived
                heuristic_note = f"≤{int(hour_limit)} h"
                break
        if candidate is None:
            dir_counts = (
                work.groupby("par_bidir", observed=True)[COL_SENDER_ID]
                .transform("nunique")
            )
            mask_pairs = dir_counts > 1
            if not mask_pairs.any():
                return pd.DataFrame(
                    columns=[
                        "timeframe",
                        "par_bidir",
                        "racha_max_yo_yo",
                        "tx_yo_yo_totales",
                        "meses_con_yo_yo",
                        "riesgo_max_par",
                        "riesgo_promedio_par",
                        "riesgo_max_yo_yo",
                        "riesgo_promedio_yo_yo",
                        "interpretabilidad",
                    ]
                )
            candidate = work.copy()
            candidate["sig_yoyo"] = mask_pairs
            heuristic_note = "sin ventana"
        work = candidate

    work = work.sort_values("fecha_hora_ts")
    work["par_bidir"] = work.apply(
        lambda row: "⇄".join(
            sorted([str(row[COL_SENDER_ID]), str(row[COL_RECEIVER_ID])])
        ),
        axis=1,
    )

    def _pair_stats(group: pd.DataFrame) -> pd.Series:
        ordered = group.sort_values("fecha_hora_ts")
        flags = ordered["sig_yoyo"].astype(bool).tolist()
        months = (
            ordered.loc[ordered["sig_yoyo"], "month_id"].dropna().astype(str).unique()
            if ordered["sig_yoyo"].any()
            else []
        )
        risk_scores = ordered.loc[ordered["sig_yoyo"], "risk_score"].astype(float)
        longest = 0
        current = 0
        total_hits = int(flags.count(True))
        for flag in flags:
            if flag:
                current += 1
                if current > longest:
                    longest = current
            else:
                current = 0
        return pd.Series(
            {
                "racha_max_yo_yo": int(longest),
                "tx_yo_yo_totales": int(total_hits),
                "meses_con_yo_yo": int(len(months)),
                "riesgo_max_yo_yo": float(risk_scores.max()) if not risk_scores.empty else 0.0,
                "riesgo_promedio_yo_yo": float(risk_scores.mean()) if not risk_scores.empty else 0.0,
            }
        )

    streaks = work.groupby("par_bidir", observed=True).apply(_pair_stats).reset_index()
    streaks = streaks[streaks["tx_yo_yo_totales"] > 0]
    if streaks.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "par_bidir",
                "racha_max_yo_yo",
                "tx_yo_yo_totales",
                "meses_con_yo_yo",
                "riesgo_max_par",
                "riesgo_promedio_par",
                "riesgo_max_yo_yo",
                "riesgo_promedio_yo_yo",
                "interpretabilidad",
            ]
        )

    pares = _get_section(reports, "par_personas", timeframe)
    if pares.empty or "pair" not in pares:
        pair_risk = pd.DataFrame(columns=["par_bidir", "riesgo_max_par", "riesgo_promedio_par"])
    else:
        tmp = pares[["pair", "risk_max", "risk_avg"]].copy()
        tmp["par_bidir"] = tmp["pair"].astype(str).apply(
            lambda text: "⇄".join(sorted(text.split("→"))) if "→" in text else text
        )
        pair_risk = (
            tmp.groupby("par_bidir", observed=True)
            .agg(
                riesgo_max_par=("risk_max", "max"),
                riesgo_promedio_par=("risk_avg", "mean"),
            )
            .reset_index()
        )

    merged = streaks.merge(pair_risk, on="par_bidir", how="left")
    merged["riesgo_max_par"] = merged["riesgo_max_par"].fillna(0.0)
    merged["riesgo_promedio_par"] = merged["riesgo_promedio_par"].fillna(0.0)

    filtered = merged[
        (merged["racha_max_yo_yo"] >= int(min_consecutive))
        & (merged["riesgo_max_par"] >= float(risk_threshold))
    ].copy()
    relaxed = False
    if filtered.empty:
        relaxed = True
        relaxed_min = max(int(min_consecutive) - 1, 1)
        candidate = merged[merged["racha_max_yo_yo"] >= relaxed_min].copy()
        if not candidate.empty:
            dynamic_threshold = candidate["riesgo_max_par"].quantile(0.6)
            if pd.isna(dynamic_threshold):
                dynamic_threshold = candidate["riesgo_max_par"].max()
            candidate = candidate[candidate["riesgo_max_par"] >= dynamic_threshold]
        if candidate.empty:
            candidate = merged.sort_values(
                ["racha_max_yo_yo", "tx_yo_yo_totales", "riesgo_max_par"],
                ascending=[False, False, False],
            ).head(25)
        filtered = candidate.copy()
        if filtered.empty:
            return pd.DataFrame(
                columns=[
                    "timeframe",
                    "par_bidir",
                    "racha_max_yo_yo",
                    "tx_yo_yo_totales",
                    "meses_con_yo_yo",
                    "riesgo_max_par",
                    "riesgo_promedio_par",
                    "riesgo_max_yo_yo",
                    "riesgo_promedio_yo_yo",
                    "interpretabilidad",
                ]
            )

    filtered["timeframe"] = timeframe
    filtered = filtered.sort_values(
        ["riesgo_max_par", "tx_yo_yo_totales"], ascending=[False, False]
    )
    filtered["interpretabilidad"] = filtered.apply(
        lambda row: (
            f"Durante '{timeframe}', el par bidireccional {row.get('par_bidir', 'sin_par')} registró "
            f"{int(row.get('tx_yo_yo_totales', 0))} transacciones clasificadas como Yo-Yo "
            f"en {int(row.get('meses_con_yo_yo', 0))} meses distintos, con una racha máxima de "
            f"{int(row.get('racha_max_yo_yo', 0))} eventos consecutivos. El riesgo máximo observado "
            f"para el par alcanzó {row.get('riesgo_max_par', 0.0):.2f} (promedio {row.get('riesgo_promedio_par', 0.0):.2f}), "
            f"mientras que las transacciones Yo-Yo llegaron a un riesgo máximo de {row.get('riesgo_max_yo_yo', 0.0):.2f}."
            + (
                " Se empleó un umbral de riesgo flexible para exponer la racha cuando el criterio "
                "estricto no devolvió casos." if relaxed and row.get("riesgo_max_par", 0.0) < float(risk_threshold) else ""
            )
            + (
                f" La racha se identificó con una heurística de ida y vuelta {heuristic_note or 'sin ventana'} ante la ausencia de banderas explícitas."
                if heuristic
                else ""
            )
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "par_bidir",
        "racha_max_yo_yo",
        "tx_yo_yo_totales",
        "meses_con_yo_yo",
        "riesgo_max_par",
        "riesgo_promedio_par",
        "riesgo_max_yo_yo",
        "riesgo_promedio_yo_yo",
        "interpretabilidad",
    ]
    return filtered.reindex(columns=columns)


def question11_near_threshold_structuring(
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
    min_months: int = 3,
    delta_limit: float = 10.0,
) -> pd.DataFrame:
    tx = _get_section(reports, "transaccion", timeframe)
    required = [
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        "month_id",
        COL_AMOUNT,
        "sig_near_thr",
        "risk_score",
    ]
    if tx.empty or not set(required).issubset(tx.columns):
        return pd.DataFrame(
            columns=[
                "timeframe",
                "pair",
                "meses_con_near",
                "tx_near_totales",
                "monto_total_near",
                "delta_promedio",
                "riesgo_max",
                "interpretabilidad",
            ]
        )

    extra_cols: list[str] = []
    if "feat_delta_near_thr" in tx.columns:
        extra_cols.append("feat_delta_near_thr")
    work = tx[required + extra_cols].copy()
    work["sig_near_thr"] = work["sig_near_thr"].fillna(False).astype(bool)
    work[COL_AMOUNT] = pd.to_numeric(work[COL_AMOUNT], errors="coerce")

    delta_estimated = "feat_delta_near_thr" not in work.columns

    def _estimate_delta(amounts: pd.Series) -> pd.Series:
        thresholds = [
            500,
            750,
            1000,
            1500,
            2000,
            3000,
            5000,
            7500,
            10000,
            15000,
            20000,
        ]
        diff = pd.DataFrame({thr: (amounts - thr).abs() for thr in thresholds})
        return diff.min(axis=1)

    if delta_estimated:
        work["feat_delta_near_thr"] = _estimate_delta(work[COL_AMOUNT].fillna(0.0))
    else:
        work["feat_delta_near_thr"] = pd.to_numeric(
            work["feat_delta_near_thr"], errors="coerce"
        )

    mask = work["sig_near_thr"] & (work["feat_delta_near_thr"] <= float(delta_limit))
    near = work.loc[mask].copy()
    relaxed_delta = False
    heuristic = False
    strategy_note: str | None = None
    if near.empty:
        relaxed_delta = True
        alt_mask = work["feat_delta_near_thr"] <= float(delta_limit)
        near = work.loc[alt_mask].copy()
        strategy_note = "delta≤limite"
    if near.empty:
        alt_mask = work["feat_delta_near_thr"] <= float(delta_limit) * 2
        near = work.loc[alt_mask].copy()
        strategy_note = "delta≤2x"
    if near.empty:
        heuristic = True
        quantile = work["feat_delta_near_thr"].quantile(0.15)
        if pd.isna(quantile) or quantile <= 0:
            quantile = work["feat_delta_near_thr"].median()
        near = work.loc[work["feat_delta_near_thr"] <= quantile].copy()
        strategy_note = f"delta≤p15 ({quantile:.2f})"
    if near.empty:
        heuristic = True
        near = work.nsmallest(50, "feat_delta_near_thr").copy()
        strategy_note = "top_deltas"
    if near.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "pair",
                "meses_con_near",
                "tx_near_totales",
                "monto_total_near",
                "delta_promedio",
                "riesgo_max",
                "interpretabilidad",
            ]
        )

    near["pair"] = near[COL_SENDER_ID].astype(str) + "→" + near[COL_RECEIVER_ID].astype(str)
    monthly = (
        near.groupby(["pair", "month_id"], observed=True)
        .agg(
            tx_near_mes=(COL_AMOUNT, "count"),
            monto_mes=(COL_AMOUNT, "sum"),
            delta_prom_mes=("feat_delta_near_thr", "mean"),
            riesgo_mes=("risk_score", "max"),
        )
        .reset_index()
    )
    agg = (
        monthly.groupby("pair", observed=True)
        .agg(
            meses_con_near=("month_id", "nunique"),
            tx_near_totales=("tx_near_mes", "sum"),
            monto_total_near=("monto_mes", "sum"),
            delta_promedio=("delta_prom_mes", "mean"),
            riesgo_max=("riesgo_mes", "max"),
        )
        .reset_index()
    )
    filtered = agg[agg["meses_con_near"] >= int(min_months)].copy()
    relaxed_months = False
    if filtered.empty:
        relaxed_months = True
        fallback_min = max(int(min_months) - 1, 1)
        filtered = agg[agg["meses_con_near"] >= fallback_min].copy()
    if filtered.empty:
        filtered = agg.sort_values(
            ["meses_con_near", "tx_near_totales", "monto_total_near"],
            ascending=[False, False, False],
        ).head(25)
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "pair",
                "meses_con_near",
                "tx_near_totales",
                "monto_total_near",
                "delta_promedio",
                "riesgo_max",
                "interpretabilidad",
            ]
        )

    filtered["timeframe"] = timeframe
    filtered = filtered.sort_values(
        ["meses_con_near", "monto_total_near"], ascending=[False, False]
    )
    filtered["interpretabilidad"] = filtered.apply(
        lambda row: (
            f"Durante '{timeframe}', el par {row.get('pair', 'sin_par')} repitió montos cerca de un umbral "
            f"en {int(row.get('meses_con_near', 0))} meses, registrando {int(row.get('tx_near_totales', 0))} "
            f"transacciones por {_format_float(row.get('monto_total_near', 0))}. La desviación promedio fue "
            f"{row.get('delta_promedio', 0.0):.2f} con riesgo máximo {row.get('riesgo_max', 0.0):.2f}."
            + (
                " Se ignoró la bandera estricta para capturar montos recurrentes pegados al umbral."
                if relaxed_delta
                else ""
            )
            + (
                " Se estimó el delta contra umbrales comunes al no contar con la métrica original."
                if delta_estimated
                else ""
            )
            + (
                " Se aplicó una heurística de selección adicional ("
                + str(strategy_note)
                + ") para aumentar la cobertura."
                if heuristic and strategy_note
                else (" Se aplicó una heurística de selección adicional." if heuristic else "")
            )
            + (
                " El número de meses proviene de un criterio flexible ante la ausencia de casos estrictos."
                if relaxed_months and row.get("meses_con_near", 0) < int(min_months)
                else ""
            )
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "pair",
        "meses_con_near",
        "tx_near_totales",
        "monto_total_near",
        "delta_promedio",
        "riesgo_max",
        "interpretabilidad",
    ]
    return filtered.reindex(columns=columns)


def question12_smurfing_chronic(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME, min_months: int = 3
) -> pd.DataFrame:
    tx = _get_section(reports, "transaccion", timeframe)
    required = [
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        "month_id",
        COL_AMOUNT,
        "sig_smurf",
        "risk_score",
    ]
    if tx.empty or not set(required).issubset(tx.columns):
        return pd.DataFrame(
            columns=[
                "timeframe",
                "pair",
                "meses_con_smurf",
                "tx_smurf_totales",
                "monto_smurf_total",
                "riesgo_promedio_mes",
                "riesgo_max",
                "tendencia_riesgo",
                "interpretabilidad",
            ]
        )

    work = tx[required].copy()
    work["sig_smurf"] = work["sig_smurf"].fillna(False).astype(bool)
    relaxed_flags = False
    smurf = work.loc[work["sig_smurf"]].copy()
    if smurf.empty:
        relaxed_flags = True
        work["pair"] = work[COL_SENDER_ID].astype(str) + "→" + work[COL_RECEIVER_ID].astype(str)
        thresholds = work.groupby("pair", observed=True)[COL_AMOUNT].transform(
            lambda s: s.quantile(0.25)
        )
        fallback_threshold = work[COL_AMOUNT].median()
        work["is_small"] = work[COL_AMOUNT] <= thresholds.fillna(fallback_threshold)
        smurf = work.loc[work["is_small"]].copy()
        smurf["sig_smurf"] = True
        if smurf.empty:
            return pd.DataFrame(
                columns=[
                    "timeframe",
                    "pair",
                    "meses_con_smurf",
                    "tx_smurf_totales",
                    "monto_smurf_total",
                    "riesgo_promedio_mes",
                    "riesgo_max",
                    "tendencia_riesgo",
                    "interpretabilidad",
                ]
            )

    smurf["pair"] = smurf[COL_SENDER_ID].astype(str) + "→" + smurf[COL_RECEIVER_ID].astype(str)
    monthly = (
        smurf.groupby(["pair", "month_id"], observed=True)
        .agg(
            tx_smurf_mes=(COL_AMOUNT, "count"),
            monto_mes=(COL_AMOUNT, "sum"),
            riesgo_prom_mes=("risk_score", "mean"),
            riesgo_max_mes=("risk_score", "max"),
        )
        .reset_index()
    )

    def _risk_trend(sub: pd.DataFrame) -> pd.Series:
        ordered = sub.copy()
        ordered["_month_sort"] = pd.to_datetime(ordered["month_id"], errors="coerce")
        ordered = ordered.sort_values(["_month_sort", "month_id"]).reset_index(drop=True)
        risk_start = float(ordered.loc[0, "riesgo_prom_mes"]) if not ordered.empty else 0.0
        risk_end = float(ordered.loc[len(ordered) - 1, "riesgo_prom_mes"]) if len(ordered) else 0.0
        trend = risk_end - risk_start
        direction = "estable"
        if trend > 0.05:
            direction = "al alza"
        elif trend < -0.05:
            direction = "a la baja"
        return pd.Series(
            {
                "riesgo_promedio_mes": float(ordered["riesgo_prom_mes"].mean()) if not ordered.empty else 0.0,
                "riesgo_max": float(ordered["riesgo_max_mes"].max()) if not ordered.empty else 0.0,
                "tendencia_riesgo": direction,
                "delta_riesgo": trend,
                "riesgo_inicio": risk_start,
                "riesgo_fin": risk_end,
            }
        )

    base = (
        monthly.groupby("pair", observed=True)
        .agg(
            meses_con_smurf=("month_id", "nunique"),
            tx_smurf_totales=("tx_smurf_mes", "sum"),
            monto_smurf_total=("monto_mes", "sum"),
        )
        .reset_index()
    )
    risk = monthly.groupby("pair", observed=True).apply(_risk_trend).reset_index()
    merged = base.merge(risk, on="pair", how="left")
    filtered = merged[merged["meses_con_smurf"] >= int(min_months)].copy()
    relaxed_months = False
    if filtered.empty:
        relaxed_months = True
        fallback_min = max(int(min_months) - 1, 1)
        filtered = merged[merged["meses_con_smurf"] >= fallback_min].copy()
    if filtered.empty:
        filtered = merged.sort_values(
            ["meses_con_smurf", "monto_smurf_total", "tx_smurf_totales"],
            ascending=[False, False, False],
        ).head(25)
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "pair",
                "meses_con_smurf",
                "tx_smurf_totales",
                "monto_smurf_total",
                "riesgo_promedio_mes",
                "riesgo_max",
                "tendencia_riesgo",
                "interpretabilidad",
            ]
        )

    filtered["timeframe"] = timeframe
    filtered = filtered.sort_values(
        ["meses_con_smurf", "monto_smurf_total"], ascending=[False, False]
    )
    filtered["interpretabilidad"] = filtered.apply(
        lambda row: (
            f"En '{timeframe}', el par {row.get('pair', 'sin_par')} mostró fraccionamiento (smurfing) en "
            f"{int(row.get('meses_con_smurf', 0))} meses, acumulando {_format_float(row.get('monto_smurf_total', 0))} "
            f"en {int(row.get('tx_smurf_totales', 0))} transacciones. El riesgo promedio mensual fue "
            f"{row.get('riesgo_promedio_mes', 0.0):.2f}, con pico de {row.get('riesgo_max', 0.0):.2f} y tendencia "
            f"{row.get('tendencia_riesgo', 'estable')} (inicio {row.get('riesgo_inicio', 0.0):.2f} → fin {row.get('riesgo_fin', 0.0):.2f})."
            + (
                " Se utilizó una heurística de montos pequeños repetidos ante la ausencia de alertas explícitas."
                if relaxed_flags and row.get("tx_smurf_totales", 0) > 0
                else ""
            )
            + (
                " Se relajó la cantidad mínima de meses para exponer un patrón persistente." if relaxed_months and row.get("meses_con_smurf", 0) < int(min_months) else ""
            )
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "pair",
        "meses_con_smurf",
        "tx_smurf_totales",
        "monto_smurf_total",
        "riesgo_promedio_mes",
        "riesgo_max",
        "tendencia_riesgo",
        "interpretabilidad",
    ]
    return filtered.reindex(columns=columns)


def question13_bad_loans_with_frequency(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME, min_months: int = 2
) -> pd.DataFrame:
    tx = _get_section(reports, "transaccion", timeframe)
    required = [
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        "month_id",
        COL_AMOUNT,
        "sig_loan_bad_repay",
        "sig_freq",
        "risk_score",
    ]
    if tx.empty or not set(required).issubset(tx.columns):
        return pd.DataFrame(
            columns=[
                "timeframe",
                "pair",
                "meses_con_coincidencia",
                "prestamos_incumplidos",
                "monto_prestamos_incumplidos",
                "eventos_alta_frecuencia",
                "riesgo_max",
                "riesgo_promedio",
                "interpretabilidad",
            ]
        )

    extra_cols: list[str] = []
    if "feat_repay_ratio" in tx.columns:
        extra_cols.append("feat_repay_ratio")
    work = tx[required + extra_cols].copy()
    work[COL_AMOUNT] = pd.to_numeric(work[COL_AMOUNT], errors="coerce")
    work["sig_loan_bad_repay"] = work["sig_loan_bad_repay"].fillna(False).astype(bool)
    work["sig_freq"] = work["sig_freq"].fillna(False).astype(bool)
    if "feat_repay_ratio" in work:
        work["feat_repay_ratio"] = pd.to_numeric(
            work["feat_repay_ratio"], errors="coerce"
        )
        work["loan_bad"] = work["sig_loan_bad_repay"] & (
            work["feat_repay_ratio"] <= 0.5
        )
    else:
        work["loan_bad"] = work["sig_loan_bad_repay"]
    work["pair"] = work[COL_SENDER_ID].astype(str) + "→" + work[COL_RECEIVER_ID].astype(str)

    heuristic = False
    strategy_note: str | None = None
    relaxed_overlap = False

    if work["loan_bad"].any():
        work["monto_loan_bad"] = work[COL_AMOUNT].where(work["loan_bad"], 0.0)
        work["freq_event"] = work["sig_freq"].astype(int)

        monthly = (
            work.groupby(["pair", "month_id"], observed=True)
            .agg(
                prestamos_incumplidos=("loan_bad", "sum"),
                monto_prestamos_incumplidos=("monto_loan_bad", "sum"),
                eventos_alta_frecuencia=("freq_event", "sum"),
                riesgo_prom_mes=("risk_score", "mean"),
                riesgo_max_mes=("risk_score", "max"),
            )
            .reset_index()
        )
        monthly["monto_prestamos_incumplidos"] = monthly["monto_prestamos_incumplidos"].fillna(0.0)
        monthly["prestamos_incumplidos"] = monthly["prestamos_incumplidos"].fillna(0).astype(int)
        monthly["eventos_alta_frecuencia"] = (
            monthly["eventos_alta_frecuencia"].fillna(0).astype(int)
        )
        monthly["coincide"] = (monthly["prestamos_incumplidos"] > 0) & (
            monthly["eventos_alta_frecuencia"] > 0
        )
        monthly["flag_prestamo"] = monthly["prestamos_incumplidos"] > 0
        monthly["flag_frecuencia"] = monthly["eventos_alta_frecuencia"] > 0
        coincidencias = monthly.loc[monthly["coincide"]].copy()
        if coincidencias.empty:
            relaxed_overlap = True
            agg = (
                monthly.groupby("pair", observed=True)
                .agg(
                    meses_prestamo=("flag_prestamo", "sum"),
                    meses_frecuencia=("flag_frecuencia", "sum"),
                    prestamos_incumplidos=("prestamos_incumplidos", "sum"),
                    monto_prestamos_incumplidos=("monto_prestamos_incumplidos", "sum"),
                    eventos_alta_frecuencia=("eventos_alta_frecuencia", "sum"),
                    riesgo_max=("riesgo_max_mes", "max"),
                    riesgo_promedio=("riesgo_prom_mes", "mean"),
                )
                .reset_index()
            )
            if agg.empty:
                return pd.DataFrame(
                    columns=[
                        "timeframe",
                        "pair",
                        "meses_con_coincidencia",
                        "prestamos_incumplidos",
                        "monto_prestamos_incumplidos",
                        "eventos_alta_frecuencia",
                        "riesgo_max",
                        "riesgo_promedio",
                        "interpretabilidad",
                    ]
                )
            agg["meses_con_coincidencia"] = agg[["meses_prestamo", "meses_frecuencia"]].min(axis=1)
            agg = agg[(agg["meses_prestamo"] > 0) & (agg["meses_frecuencia"] > 0)].copy()
            agg = agg.drop(columns=["meses_prestamo", "meses_frecuencia"], errors="ignore")
        else:
            agg = (
                coincidencias.groupby("pair", observed=True)
                .agg(
                    meses_con_coincidencia=("month_id", "nunique"),
                    prestamos_incumplidos=("prestamos_incumplidos", "sum"),
                    monto_prestamos_incumplidos=("monto_prestamos_incumplidos", "sum"),
                    eventos_alta_frecuencia=("eventos_alta_frecuencia", "sum"),
                    riesgo_max=("riesgo_max_mes", "max"),
                    riesgo_promedio=("riesgo_prom_mes", "mean"),
                )
                .reset_index()
            )
    else:
        heuristic = True
        work["par_bidir"] = work.apply(
            lambda row: "⇄".join(sorted([str(row[COL_SENDER_ID]), str(row[COL_RECEIVER_ID])])),
            axis=1,
        )

        def _pair_month_metrics(group: pd.DataFrame) -> pd.Series:
            if group.empty:
                return pd.Series(dtype=float)
            totals = (
                group.groupby([COL_SENDER_ID, COL_RECEIVER_ID])[COL_AMOUNT]
                .agg(["sum", "count"])
                .rename(columns={"sum": "amount_sum", "count": "count_tx"})
            )
            if totals.empty:
                return pd.Series(dtype=float)
            loan_key = totals["amount_sum"].idxmax()
            loan_sender, loan_receiver = loan_key
            loan_sum = float(totals.loc[loan_key, "amount_sum"])
            loan_count = int(totals.loc[loan_key, "count_tx"])
            repay_key = (loan_receiver, loan_sender)
            repay_sum = float(totals.loc[repay_key, "amount_sum"]) if repay_key in totals.index else 0.0
            repay_count = int(totals.loc[repay_key, "count_tx"]) if repay_key in totals.index else 0
            repay_ratio = repay_sum / loan_sum if loan_sum else 0.0
            loss_amount = max(loan_sum - repay_sum, 0.0)
            total_events = int(group.shape[0])
            risk_max = float(group["risk_score"].max()) if "risk_score" in group else 0.0
            risk_avg = float(group["risk_score"].mean()) if "risk_score" in group else 0.0
            return pd.Series(
                {
                    "pair": f"{loan_sender}→{loan_receiver}",
                    "month_id": group["month_id"].iloc[0],
                    "prestamos_incumplidos": int(loan_count),
                    "monto_prestamos_incumplidos": float(loss_amount),
                    "eventos_alta_frecuencia": total_events,
                    "riesgo_max_mes": risk_max,
                    "riesgo_prom_mes": risk_avg,
                    "repay_ratio": float(repay_ratio),
                }
            )

        monthly = (
            work.groupby(["par_bidir", "month_id"], observed=True)
            .apply(_pair_month_metrics)
            .dropna()
            .reset_index(drop=True)
        )
        if monthly.empty:
            return pd.DataFrame(
                columns=[
                    "timeframe",
                    "pair",
                    "meses_con_coincidencia",
                    "prestamos_incumplidos",
                    "monto_prestamos_incumplidos",
                    "eventos_alta_frecuencia",
                    "riesgo_max",
                    "riesgo_promedio",
                    "interpretabilidad",
                ]
            )
        freq_threshold = monthly["eventos_alta_frecuencia"].quantile(0.75)
        if pd.isna(freq_threshold) or freq_threshold < 3:
            freq_threshold = 3
        monthly["flag_freq"] = monthly["eventos_alta_frecuencia"] >= freq_threshold
        monthly["flag_prestamo"] = monthly["repay_ratio"] <= 0.5
        coincidencias = monthly.loc[monthly["flag_freq"] & monthly["flag_prestamo"]].copy()
        strategy_note = f"freq≥{int(freq_threshold)} y ratio≤0.50"
        if coincidencias.empty:
            freq_threshold = max(2, monthly["eventos_alta_frecuencia"].quantile(0.6))
            if pd.isna(freq_threshold):
                freq_threshold = 2
            monthly["flag_freq"] = monthly["eventos_alta_frecuencia"] >= freq_threshold
            coincidencias = monthly.loc[monthly["flag_freq"] & monthly["flag_prestamo"]].copy()
            strategy_note = f"freq≥{int(freq_threshold)} y ratio≤0.50 (flexible)"
        if coincidencias.empty:
            relaxed_overlap = True
            coincidencias = monthly.sort_values(
                ["eventos_alta_frecuencia", "monto_prestamos_incumplidos"],
                ascending=[False, False],
            ).head(50)
        agg = (
            coincidencias.groupby("pair", observed=True)
            .agg(
                meses_con_coincidencia=("month_id", "nunique"),
                prestamos_incumplidos=("prestamos_incumplidos", "sum"),
                monto_prestamos_incumplidos=("monto_prestamos_incumplidos", "sum"),
                eventos_alta_frecuencia=("eventos_alta_frecuencia", "sum"),
                riesgo_max=("riesgo_max_mes", "max"),
                riesgo_promedio=("riesgo_prom_mes", "mean"),
            )
            .reset_index()
        )

    if agg.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "pair",
                "meses_con_coincidencia",
                "prestamos_incumplidos",
                "monto_prestamos_incumplidos",
                "eventos_alta_frecuencia",
                "riesgo_max",
                "riesgo_promedio",
                "interpretabilidad",
            ]
        )

    agg["prestamos_incumplidos"] = agg["prestamos_incumplidos"].fillna(0).astype(int)
    agg["eventos_alta_frecuencia"] = agg["eventos_alta_frecuencia"].fillna(0).astype(int)
    filtered = agg[agg["meses_con_coincidencia"] >= int(min_months)].copy()
    relaxed_months = False
    if filtered.empty:
        relaxed_months = True
        fallback_min = max(int(min_months) - 1, 1)
        filtered = agg[agg["meses_con_coincidencia"] >= fallback_min].copy()
    if filtered.empty:
        filtered = agg.sort_values(
            ["meses_con_coincidencia", "monto_prestamos_incumplidos", "eventos_alta_frecuencia"],
            ascending=[False, False, False],
        ).head(25)
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "pair",
                "meses_con_coincidencia",
                "prestamos_incumplidos",
                "monto_prestamos_incumplidos",
                "eventos_alta_frecuencia",
                "riesgo_max",
                "riesgo_promedio",
                "interpretabilidad",
            ]
        )

    filtered["timeframe"] = timeframe
    filtered = filtered.sort_values(
        ["meses_con_coincidencia", "monto_prestamos_incumplidos"], ascending=[False, False]
    )
    filtered["interpretabilidad"] = filtered.apply(
        lambda row: (
            f"Durante '{timeframe}', el par {row.get('pair', 'sin_par')} combinó préstamos con repago ≤50% "
            f"y ráfagas de ≥5 transacciones/30 días en {int(row.get('meses_con_coincidencia', 0))} meses. "
            f"Se detectaron {int(row.get('prestamos_incumplidos', 0))} préstamos incumplidos por "
            f"{_format_float(row.get('monto_prestamos_incumplidos', 0))} y {int(row.get('eventos_alta_frecuencia', 0))} eventos"
            f"de alta frecuencia, con riesgo máximo {row.get('riesgo_max', 0.0):.2f}."
            + (
                " Se utilizó una heurística para aproximar préstamos incumplidos y ráfagas "
                + (f"({strategy_note})." if strategy_note else ".")
                if heuristic
                else ""
            )
            + (
                " Los patrones se observaron de forma flexible aun cuando no coincidieron exactamente en el mismo mes."
                if relaxed_overlap
                else ""
            )
            + (
                " Se redujo el requisito estricto de meses para resaltar la reiteración de la conducta."
                if relaxed_months and row.get("meses_con_coincidencia", 0) < int(min_months)
                else ""
            )
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "pair",
        "meses_con_coincidencia",
        "prestamos_incumplidos",
        "monto_prestamos_incumplidos",
        "eventos_alta_frecuencia",
        "riesgo_max",
        "riesgo_promedio",
        "interpretabilidad",
    ]
    return filtered.reindex(columns=columns)


def question14_recurrent_payroll(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME, min_months: int = 3
) -> pd.DataFrame:
    tx = _get_section(reports, "transaccion", timeframe)
    required = [
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        "fecha_hora_ts",
        COL_AMOUNT,
        "sig_recurrent",
    ]
    if tx.empty or not set(required).issubset(tx.columns):
        return pd.DataFrame(
            columns=[
                "timeframe",
                "emisor",
                "receptor",
                "dia_corte",
                "meses_recurrentes",
                "tx_totales",
                "monto_total",
                "monto_promedio",
                "interpretabilidad",
            ]
        )

    work = tx[required].copy()
    work["sig_recurrent"] = work["sig_recurrent"].fillna(False).astype(bool)
    relaxed_flags = False
    if not work["sig_recurrent"].any():
        relaxed_flags = True

    ts = work["fecha_hora_ts"].dt.tz_convert(None)
    work["dia_corte"] = ts.dt.day
    work["mes_periodo"] = ts.dt.to_period("M").astype(str)

    grouped = (
        work.groupby([COL_SENDER_ID, COL_RECEIVER_ID, "dia_corte"], observed=True)
        .agg(
            meses_recurrentes=("mes_periodo", "nunique"),
            tx_totales=(COL_AMOUNT, "count"),
            monto_total=(COL_AMOUNT, "sum"),
            monto_promedio=(COL_AMOUNT, "mean"),
            flag_recurrent=("sig_recurrent", "mean"),
        )
        .reset_index()
    )

    filtered = grouped[
        (grouped["meses_recurrentes"] >= int(min_months))
        & ((grouped["flag_recurrent"] > 0) | relaxed_flags)
    ].copy()
    relaxed_months = False
    if filtered.empty:
        relaxed_months = True
        fallback_min = max(int(min_months) - 1, 1)
        filtered = grouped[
            (grouped["meses_recurrentes"] >= fallback_min)
        ].copy()
    if filtered.empty:
        filtered = grouped.sort_values(
            ["meses_recurrentes", "monto_total", "tx_totales"],
            ascending=[False, False, False],
        ).head(25)
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "emisor",
                "receptor",
                "dia_corte",
                "meses_recurrentes",
                "tx_totales",
                "monto_total",
                "monto_promedio",
                "interpretabilidad",
            ]
        )

    filtered["timeframe"] = timeframe
    filtered.rename(
        columns={COL_SENDER_ID: "emisor", COL_RECEIVER_ID: "receptor"}, inplace=True
    )
    filtered = filtered.sort_values(
        ["meses_recurrentes", "monto_total"], ascending=[False, False]
    )
    filtered["interpretabilidad"] = filtered.apply(
        lambda row: (
            f"En '{timeframe}', el emisor {row.get('emisor', 'sin_emisor')} pagó de forma mensual recurrente al receptor "
            f"{row.get('receptor', 'sin_receptor')} durante {int(row.get('meses_recurrentes', 0))} meses consecutivos, "
            f"siempre cerca del día {int(row.get('dia_corte', 0))}. Se identificaron {int(row.get('tx_totales', 0))} "
            f"pagos por un total de {_format_float(row.get('monto_total', 0))} y un promedio mensual de "
            f"{_format_float(row.get('monto_promedio', 0))}, lo que sugiere una nómina o compensación paralela."
            + (
                " Se relajó la exigencia de bandera recurrente para resaltar pagos de calendario repetidos."
                if relaxed_flags and row.get("flag_recurrent", 0) == 0
                else ""
            )
            + (
                " Se redujo el mínimo de meses ante la ausencia de coincidencias estrictas."
                if relaxed_months and row.get("meses_recurrentes", 0) < int(min_months)
                else ""
            )
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "emisor",
        "receptor",
        "dia_corte",
        "meses_recurrentes",
        "tx_totales",
        "monto_total",
        "monto_promedio",
        "interpretabilidad",
    ]
    return filtered.reindex(columns=columns)


def _run_questions(reports: Mapping[str, Any], timeframe: str) -> Dict[str, Any]:
    return {
        "q1_manager_nlp": question1_manager_nlp(reports, timeframe),
        "q2_manager_concepts": question2_manager_concepts(reports, timeframe),
        "q3_quid_pairs": question3_quid_pairs(reports, timeframe),
        "q4_quid_negative_value_vs_load": question4_quid_negative_value_vs_load(reports, timeframe),
        "q5_reference_reuse": question5_reference_reuse(reports, timeframe),
        "q6_centralizers": question6_centralizers(reports, timeframe),
        "q7_net_imbalance": question7_net_imbalance(reports, timeframe),
        "q8_case13_new_employees": question8_case13_new_employees(reports, timeframe),
        "q9_case14_veterans_from_newcomers": question9_case14_veterans_from_newcomers(
            reports, timeframe
        ),
        "q10_yoyo_streaks": question10_yoyo_streaks(reports, timeframe),
        "q11_near_threshold_structuring": question11_near_threshold_structuring(
            reports, timeframe
        ),
        "q12_smurfing_chronic": question12_smurfing_chronic(reports, timeframe),
        "q13_bad_loans_with_frequency": question13_bad_loans_with_frequency(
            reports, timeframe
        ),
        "q14_recurrent_payroll": question14_recurrent_payroll(reports, timeframe),
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
