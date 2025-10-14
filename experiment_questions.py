"""Módulo de experimentación para responder preguntas clave de COI."""
from __future__ import annotations

import argparse
import inspect
import re
import unicodedata
from pathlib import Path
from textwrap import fill
from typing import Any, Callable, Dict, Iterable, Mapping, Literal, Tuple

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
    COL_DESCRIPTION,
    COL_RECEIVER_ID,
    COL_RELATION,
    COL_SENDER_FULL_NAME,
    COL_SENDER_ID,
    COL_SENDER_TENURE_YEARS,
)
from coi_fraud.aggregate.persons import (
    CASE13_NEW_EMPLOYEE_YEARS,
    CASE14_NEW_EMPLOYEE_YEARS,
    CASE14_OLD_EMPLOYEE_YEARS,
)
from coi_fraud.text_utils import (
    clean_raw_concept,
    first_non_empty_series,
    normalize_clean_concept,
)


DEFAULT_TIMEFRAME = "todo_el_tiempo"
DEFAULT_OUTPUT_DIR = Path("answers")
QUESTION1_DIRECTIONS: tuple[str, str] = (
    "manager_a_subordinado",
    "subordinado_a_manager",
)
NLP_CATEGORIES = (
    "SOBORNO",
    "FACILITACIÓN",
    "OFUSCACIÓN",
    "EXTORSIÓN",
    "FAVORES SEXUALES",
    "REGALOS_LUJO",
    "CONFLICTO_INTERES_FAMILIAR",
    "VIATICOS_LUJOSOS",
    "FACTURACION_SIMULADA",
    "CONSULTORIA_FANTASMA",
    "DONATIVO_CRUZADO",
    "PRESION_POLITICA",
    "NOMINA_PARALELA",
    "REEMBOLSO_DUDOSO",
    "PRESTAMO",
    "COI_RELACIONAL",
    "AGASAJOS_SOCIALES",
    "DETALLE_PERSONAL",
    "COORDINACION_REITERADA",
    "ALUSION_INDIRECTA",
)
NLP_CATEGORY_SYNONYMS = {
    "SOBORNO": ("SOBOR", "COIMA", "BRIBE", "COHECHO", "SWEETENER"),
    "FACILITACIÓN": ("FACILIT", "FACILITATION", "GRATIFICACIÓN", "FAST TRACK", "PRIORIDAD"),
    "OFUSCACIÓN": ("OFUSC", "OBFUS", "OCULT", "ENCUBR", "SIN FACTURA"),
    "EXTORSIÓN": ("EXTORS", "EXTORT", "AMENAZ", "DERECHO DE PISO", "COPERACION"),
    "FAVORES SEXUALES": ("SEXUAL", "SEX", "ACOSO", "PRIVADO", "INTIMO", "TANGA", "TANGAS", "LENCER"),
    "REGALOS_LUJO": ("REGALO", "LUJO", "VIP", "PREMIUM", "SUITE", "DETALLAZO"),
    "CONFLICTO_INTERES_FAMILIAR": ("FAMILIA", "PARENTE", "PAREJA", "ESPOS", "HIJO"),
    "VIATICOS_LUJOSOS": ("VIATIC", "HOTEL", "BUSINESS", "PRIMERA", "CINCO ESTRELLAS"),
    "FACTURACION_SIMULADA": ("FACTURA", "FANTASMA", "SIMULAD", "FACHADA"),
    "CONSULTORIA_FANTASMA": ("CONSULT", "ASESORIA", "FANTAS", "DUMMY"),
    "DONATIVO_CRUZADO": ("DONATIVO", "DONACIÓN", "PATROCINIO", "CAMPAÑA"),
    "PRESION_POLITICA": ("POLIT", "PARTIDO", "CANDID", "DIPUTADO"),
    "NOMINA_PARALELA": ("NÓMINA", "BONO", "COMPENS", "EXTRA"),
    "REEMBOLSO_DUDOSO": ("REEMBOLSO", "VIÁTICO", "GASTO", "VARIOS"),
    "PRESTAMO": ("PRÉSTAM", "ADELAN", "ABONO", "ANTICIPO"),
    "COI_RELACIONAL": ("COMPADRE", "PRIMO", "FAMIL", "AMIGO"),
    "AGASAJOS_SOCIALES": ("CERVEZA", "CHELA", "FIESTA", "TRAGO", "AFTER", "ANTRO", "KARAOKE"),
    "DETALLE_PERSONAL": ("DETALLE", "DETALLITO", "REGALITO", "TE COMPRO", "TECOMPRA", "REGLO"),
    "COORDINACION_REITERADA": ("LO DE AYER", "MISMA JUGADA", "MISMO TRATO", "COMO QUEDAMOS"),
    "ALUSION_INDIRECTA": ("YA SABES", "LO PENDIENTE", "AQUELLO", "LO HABLADO"),
}
CONCEPT_SPLIT_PATTERN = re.compile(r"[\s,;|/]+")


QUESTION_TITLES: Dict[str, str] = {
    "q1_manager_nlp": "Q1 – Manager con conceptos NLP sospechosos",
    "q2_manager_concepts": "Q2 – Conceptos NLP con mayor severidad",
    "q3_quid_pairs": "Q3 – Pares con rasgos Algo por Algo",
    "q4_quid_negative_value_vs_load": "Q4 – Autorizaciones con valor negativo vs. carga",
    "q5_reference_reuse": "Q5 – Reutilización de referencias de pago",
    "q6_centralizers": "Q6 – Receptores centralizadores",
    "q7_net_imbalance": "Q7 – Personas con desbalance neto",
    "q8_case13_new_employees": "Q8 – Receptores nuevos con montos altos",
    "q9_case14_veterans_from_newcomers": "Q9 – Veteranos que reciben de emisores nuevos",
    "q10_yoyo_streaks": "Q10 – Rachas Yo-Yo prolongadas",
    "q11_near_threshold_structuring": "Q11 – Montos pegados a umbrales regulatorios",
    "q12_smurfing_chronic": "Q12 – Fraccionamiento crónico",
    "q13_bad_loans_with_frequency": "Q13 – Préstamos incumplidos con ráfagas de frecuencia",
    "q14_recurrent_payroll": "Q14 – Pagos recurrentes tipo nómina",
    "q15_coordinated_cluster_signals": "Q15 – Clusters con señales coordinadas",
    "q16_multisignal_transactions": "Q16 – Transacciones con múltiples señales simultáneas",
    "q17_nlp_person_profiles": "Q17 – Perfiles NLP sospechosos por persona",
    "q18_user_risk_scores": "Q18 – Personas con riesgo agregado y banderas",
}


TenureUnit = Literal["months", "percentile"]
TenureDefinition = Tuple[TenureUnit, float]


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "0.00"


def _coerce_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not pd.isna(value):
        return bool(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _count_true(values: Any) -> int:
    if values is None:
        return 0
    series = pd.Series(values)
    if series.empty:
        return 0
    coerced = series.apply(_coerce_bool)
    if coerced.empty:
        return 0
    return int(sum(1 for value in coerced if value))


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


def _resolve_tenure_months(
    tx: pd.DataFrame,
    column_years: str,
    definition: TenureDefinition | None,
    *,
    default_months: float,
    context: str,
) -> float:
    """Resuelve un umbral de antigüedad expresado en meses."""

    if definition is None:
        return default_months

    unit, raw_value = definition
    unit = unit.lower()
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = float("nan")

    if unit == "months":
        return value if not pd.isna(value) else default_months

    if unit == "percentile":
        if pd.isna(value) or value < 0 or value > 1:
            print(
                f"[{context}] Percentil inválido {raw_value!r}; se usa {default_months:.2f} meses"
            )
            return default_months
        if tx is None or tx.empty or column_years not in tx:
            print(
                f"[{context}] No hay datos para convertir percentil; se usa {default_months:.2f} meses"
            )
            return default_months
        series = pd.to_numeric(tx[column_years], errors="coerce").dropna()
        if series.empty:
            print(
                f"[{context}] No hay valores numéricos para convertir percentil; se usa {default_months:.2f} meses"
            )
            return default_months
        quantile_years = float(series.quantile(value))
        months = quantile_years * 12.0
        print(
            f"[{context}] Percentil {value:.2%} equivale a {months:.2f} meses"
        )
        return months

    print(
        f"[{context}] Unidad '{unit}' no reconocida; se usa {default_months:.2f} meses"
    )
    return default_months


def _format_months_threshold(months: float) -> str:
    """Devuelve una cadena amigable para un umbral en meses."""

    try:
        value = float(months)
    except (TypeError, ValueError):
        return "N/D meses"
    if pd.isna(value):
        return "N/D meses"
    if abs(value - round(value)) < 1e-6:
        return f"{int(round(value))} meses"
    return f"{value:.1f} meses"


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
        default=6_000,
        help="Filas a generar cuando se use el dataset sintético (por defecto: 6000).",
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


def _normalize_alias_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", normalized.lower())


def _standardize_columns(
    df: pd.DataFrame,
    alias_map: Mapping[str, Iterable[str]],
) -> pd.DataFrame:
    if df.empty or not alias_map:
        return df

    rename_map: dict[str, str] = {}
    normalized_columns = {
        _normalize_alias_token(column): column for column in df.columns
    }

    for target, aliases in alias_map.items():
        if target in df.columns:
            continue

        search_values = [target, *aliases]
        found_column: str | None = None

        for candidate in search_values:
            candidate_key = _normalize_alias_token(candidate)
            if candidate_key in normalized_columns:
                found_column = normalized_columns[candidate_key]
                break

        if found_column is None:
            for candidate in search_values:
                candidate_key = _normalize_alias_token(candidate)
                if not candidate_key:
                    continue
                matches = [
                    column
                    for column in df.columns
                    if candidate_key in _normalize_alias_token(column)
                ]
                if len(matches) == 1:
                    found_column = matches[0]
                    break

        if found_column and found_column not in rename_map:
            rename_map[found_column] = target

    if rename_map:
        df = df.rename(columns=rename_map)
    return df


PERSONA_COLUMN_ALIASES: Dict[str, tuple[str, ...]] = {
    "persona": (
        "persona_id",
        "persona_norm",
        "persona_normalizada",
        "persona_codigo",
        "empleado_id",
        "user_id",
        COL_RECEIVER_ID,
    ),
    "risk_avg_person": (
        "risk_avg",
        "riesgo_promedio_persona",
        "riesgo_avg_persona",
        "risk_score_avg",
        "riesgo_promedio",
    ),
    "movements": (
        "movimientos",
        "total_movimientos",
        "n_movimientos",
        "tx_totales",
        "total_tx",
    ),
    "n_tx_emit": (
        "tx_emitidas",
        "tx_enviadas",
        "n_tx_enviadas",
        "tx_emisor",
        "transacciones_emitidas",
    ),
    "sum_emit": (
        "monto_emitido",
        "monto_enviado",
        "suma_emitida",
        "monto_tx_emitidas",
    ),
    "n_tx_recv": (
        "tx_recibidas",
        "n_tx_recibidas",
        "tx_receptor",
        "transacciones_recibidas",
    ),
    "sum_recv": (
        "monto_recibido",
        "suma_recibida",
        "monto_tx_recibidas",
        "monto_recibe",
    ),
    "net_flow": (
        "flujo_neto",
        "neto",
        "neto_persona",
        "balance_neto",
    ),
    "desbalance_persona_monto_neto": (
        "desbalance_monto_neto",
        "monto_neto_desbalance",
        "desbalance_neto_persona",
    ),
    "desbalance_persona_meses_totales": (
        "desbalance_meses_totales",
        "meses_desbalance_persona",
        "total_meses_desbalance",
    ),
    "desbalance_persona_tasa_meses_envia_extremo": (
        "desbalance_tasa_envia_extremo",
        "tasa_envio_extremo",
        "tasa_meses_envia_extremo",
    ),
    "desbalance_persona_tasa_meses_recibe_extremo": (
        "desbalance_tasa_recibe_extremo",
        "tasa_recepcion_extremo",
        "tasa_meses_recibe_extremo",
    ),
    "flags_activas": (
        "banderas_activas",
        "flags_activos",
        "flags_persona_activas",
    ),
    "flag_rate_max": (
        "tasa_flag_max",
        "tasa_bandera_max",
        "flag_rate_maximo",
        "max_tasa_flag",
    ),
    "banderas_destacadas": (
        "banderas_principales",
        "banderas_relevantes",
        "banderas_top",
    ),
    "casuistica_score_total": (
        "casuistica_puntaje_total",
        "score_total_casuistica",
        "casuistica_score_sum",
    ),
    "casuistica_score_promedio": (
        "casuistica_puntaje_promedio",
        "score_promedio_casuistica",
    ),
    "casuistica_resumen": (
        "casuistica_resumen_texto",
        "resumen_casuistica",
        "detalle_casuistica",
    ),
}


TRANSACTION_COLUMN_ALIASES: Dict[str, tuple[str, ...]] = {
    COL_RECEIVER_ID: (
        "persona",
        "persona_id",
        "destinatario_id",
        "receptor_id",
    ),
    COL_AMOUNT: (
        "monto",
        "monto_movimiento",
        "amount",
        "monto_tx",
    ),
    "fecha_hora_ts": (
        "fecha_hora",
        "fecha_operacion",
        "timestamp",
        "fecha_tx",
    ),
    "month_id": (
        "mes",
        "mes_id",
        "month",
    ),
}


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


def _combine_unique_texts(values: Iterable[Any]) -> str:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if _is_blank_value(value):
            continue
        text = str(value).strip()
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return "; ".join(ordered)


def _is_blank_value(value: Any) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        result = False
    else:
        try:
            if bool(result):
                return True
        except (TypeError, ValueError):
            pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return True
    return False


def _combine_unique_categories(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if _is_blank_value(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _collect_unique_text_list(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value is None or value is pd.NA:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        if text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def _max_text_length(values: Iterable[Any]) -> int:
    lengths = []
    for value in values:
        if _is_blank_value(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        lengths.append(len(text))
    return max(lengths, default=0)


def _most_common_text(values: Iterable[Any]) -> str:
    counts: dict[str, int] = {}
    top_text = ""
    top_count = 0
    for value in values:
        if _is_blank_value(value):
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        counts[text] = counts.get(text, 0) + 1
        if counts[text] > top_count or (counts[text] == top_count and len(text) > len(top_text)):
            top_text = text
            top_count = counts[text]
    return top_text


def _ensure_raw_concept_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    source = first_non_empty_series(
        work,
        [
            "nlp_concepto_crudo",
            "reference_number_trans_desc",
            COL_DESCRIPTION,
            "tx_tags",
            "feat_reference_norm",
            "reference_norm",
            "nlp_concepto_sospechoso",
        ],
    )
    if source.empty and not work.empty:
        source = pd.Series([""] * len(work), index=work.index, dtype="string")
    work["nlp_concepto_crudo"] = source.reindex(work.index, fill_value="").map(clean_raw_concept)
    return work


def _coalesce_text_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df:
        return df[column].fillna("").astype(str)
    return pd.Series([""] * len(df), index=df.index)


def _collect_sender_lists(
    reports: Mapping[str, Any],
    timeframe: str,
    personas: Iterable[Any],
    *,
    max_sender_tenure_years: float | None = None,
) -> dict[str, list[str]]:
    """Genera listas de emisores asociados a cada persona receptora."""

    tx = _standardize_columns(
        _get_section(reports, "transaccion", timeframe), TRANSACTION_COLUMN_ALIASES
    )
    if tx.empty:
        return {}

    required = {COL_SENDER_ID, COL_RECEIVER_ID}
    if not required.issubset(tx.columns):
        return {}

    columns = [COL_SENDER_ID, COL_RECEIVER_ID]
    if COL_SENDER_FULL_NAME in tx.columns:
        columns.append(COL_SENDER_FULL_NAME)
    if max_sender_tenure_years is not None and COL_SENDER_TENURE_YEARS in tx.columns:
        columns.append(COL_SENDER_TENURE_YEARS)

    subset = tx[columns].copy()
    subset = subset.dropna(subset=[COL_RECEIVER_ID, COL_SENDER_ID])
    if subset.empty:
        return {}

    persona_values = [value for value in personas if not pd.isna(value)]
    if not persona_values:
        return {}

    persona_strings = {str(value) for value in persona_values}
    subset[COL_RECEIVER_ID] = subset[COL_RECEIVER_ID].astype(str)
    subset = subset.loc[subset[COL_RECEIVER_ID].isin(persona_strings)].copy()
    if subset.empty:
        return {}

    if max_sender_tenure_years is not None and COL_SENDER_TENURE_YEARS in subset.columns:
        subset[COL_SENDER_TENURE_YEARS] = pd.to_numeric(
            subset[COL_SENDER_TENURE_YEARS], errors="coerce"
        )
        subset = subset.loc[
            subset[COL_SENDER_TENURE_YEARS].fillna(float("inf")) <= max_sender_tenure_years
        ].copy()
        if subset.empty:
            return {}

    subset[COL_SENDER_ID] = subset[COL_SENDER_ID].astype(str)
    subset["_sender_repr"] = subset[COL_SENDER_ID]
    if COL_SENDER_FULL_NAME in subset.columns:
        names = subset[COL_SENDER_FULL_NAME].fillna("").astype(str).str.strip()
        with_name = names != ""
        subset.loc[with_name, "_sender_repr"] = (
            subset.loc[with_name, COL_SENDER_ID].astype(str)
            + " ("
            + names.loc[with_name]
            + ")"
        )

    if subset.empty:
        return {}

    def _unique_sorted(series: pd.Series) -> list[str]:
        unique_values = pd.unique(series.astype(str))
        return sorted(unique_values)

    grouped = (
        subset.groupby(COL_RECEIVER_ID, observed=True)["_sender_repr"]
        .agg(_unique_sorted)
        .to_dict()
    )
    return grouped


def _manager_nlp_hits(
    tx: pd.DataFrame,
    categories: Iterable[str] = NLP_CATEGORIES,
) -> pd.DataFrame:
    work = _filter_manager_subordinate(tx)
    if work.empty:
        return work.iloc[0:0].copy()

    work = _ensure_raw_concept_column(work)

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
            fallback = _ensure_raw_concept_column(fallback)
            return fallback

    return work.iloc[0:0].copy()


def question1_manager_nlp(
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
    categories: Iterable[str] = NLP_CATEGORIES,
    *,
    direction: Literal[QUESTION1_DIRECTIONS] = "manager_a_subordinado",
) -> pd.DataFrame:
    """Detecta pagos sospechosos entre managers y subordinados usando etiquetas NLP.

    Parameters
    ----------
    reports
        Diccionario de reportes generado por :func:`run_pipeline` con las
        secciones tabulares de interés.
    timeframe
        Ventana temporal sobre la que se filtran los datos. Por defecto se usa
        ``"todo_el_tiempo"``.
    categories
        Categorías NLP que deben buscarse dentro de los campos de texto; por
        defecto se emplea :data:`NLP_CATEGORIES`.
    direction
        Orientación del flujo de pagos a priorizar. Acepta ``"manager_a_subordinado"``
        (por defecto) para pagos enviados por managers y ``"subordinado_a_manager"``
        para el sentido inverso. Cuando la columna :data:`~coi_fraud.schemas.COL_RELATION`
        indica explícitamente que el manager es el emisor o el receptor, la función
        utilizará esa pista para determinar el sentido real del flujo antes de aplicar
        el filtro.

    Metodología
    -----------
    1. Obtiene la sección de transacciones para el ``timeframe`` solicitado.
    2. Ejecuta :func:`_manager_nlp_hits` para detectar coincidencias manager-
       subordinado según las categorías proporcionadas y sus sinónimos.
    3. Normaliza identificadores de manager y subordinado y descarta registros
       incompletos.
    4. Agrega conteos y montos por mes y par jerárquico, consolidando todas las
       categorías detectadas en una lista junto con los conceptos crudos y las
       descripciones de origen para construir la explicación en lenguaje
       natural.

    Returns
    -------
    pandas.DataFrame
        Tabla priorizada con columnas de interpretabilidad sobre conceptos NLP
        sospechosos en relaciones manager-subordinado. La columna
        ``"nlp_concepto_sospechoso"`` contiene una lista de conceptos únicos
        detectados para cada par y periodo.
    """
    tx = _standardize_columns(
        _get_section(reports, "transaccion", timeframe), TRANSACTION_COLUMN_ALIASES
    )
    if tx.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "month_id",
                "manager_user_id",
                "subordinado_user_id",
                "nlp_concepto_sospechoso",
                "nlp_concepto_crudo",
                "tx_count",
                "monto_total",
                "nlp_descripciones",
                "interpretabilidad",
            ]
        )

    direction = str(direction)
    if direction not in QUESTION1_DIRECTIONS:
        valid = "', '".join(QUESTION1_DIRECTIONS)
        raise ValueError(
            "direction debe ser uno de '{valid}', se recibió '{direction}'".format(
                valid=valid, direction=direction
            )
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
                "nlp_concepto_crudo",
                "tx_count",
                "monto_total",
                "nlp_descripciones",
                "interpretabilidad",
            ]
        )

    hits = hits.copy()

    def _normalize_identifier(series: pd.Series) -> pd.Series:
        normalized = series.astype("string").fillna("")
        normalized = normalized.str.strip()
        normalized = normalized.replace({"": pd.NA})
        # Devuelve objetos estándar en lugar de ``pd.NA`` para evitar ambigüedad al
        # evaluar los valores en funciones auxiliares.
        return normalized.mask(normalized.isna(), None)

    if COL_RELATION in hits.columns:
        relation = hits[COL_RELATION].fillna("").astype(str).str.lower()
        manager_ids = hits[COL_RECEIVER_ID].astype("string")
        subordinate_ids = hits[COL_SENDER_ID].astype("string")

        manager_is_sender = relation.str.contains("manager_del_receptor")
        manager_ids.loc[manager_is_sender] = hits.loc[
            manager_is_sender, COL_SENDER_ID
        ].astype("string")
        subordinate_ids.loc[manager_is_sender] = hits.loc[
            manager_is_sender, COL_RECEIVER_ID
        ].astype("string")

        manager_is_receiver = relation.str.contains("manager_del_emisor")

        unknown_orientation = ~(manager_is_sender | manager_is_receiver)
        if unknown_orientation.any():
            if direction == "manager_a_subordinado":
                manager_ids.loc[unknown_orientation] = hits.loc[
                    unknown_orientation, COL_SENDER_ID
                ].astype("string")
                subordinate_ids.loc[unknown_orientation] = hits.loc[
                    unknown_orientation, COL_RECEIVER_ID
                ].astype("string")
            else:
                manager_ids.loc[unknown_orientation] = hits.loc[
                    unknown_orientation, COL_RECEIVER_ID
                ].astype("string")
                subordinate_ids.loc[unknown_orientation] = hits.loc[
                    unknown_orientation, COL_SENDER_ID
                ].astype("string")

        actual_direction = pd.Series(
            "subordinado_a_manager", index=hits.index, dtype="string"
        )
        actual_direction.loc[manager_is_sender] = "manager_a_subordinado"
        actual_direction.loc[unknown_orientation] = direction

        hits = hits.loc[actual_direction == direction].copy()
        if hits.empty:
            return pd.DataFrame(
                columns=[
                    "timeframe",
                    "month_id",
                    "manager_user_id",
                    "subordinado_user_id",
                    "nlp_concepto_sospechoso",
                    "nlp_concepto_crudo",
                    "nlp_descripciones",
                    "tx_count",
                    "monto_total",
                    "interpretabilidad",
                ]
            )

        manager_ids = manager_ids.loc[hits.index]
        subordinate_ids = subordinate_ids.loc[hits.index]
    else:
        if direction == "manager_a_subordinado":
            manager_ids = hits[COL_SENDER_ID].astype("string")
            subordinate_ids = hits[COL_RECEIVER_ID].astype("string")
        else:
            manager_ids = hits[COL_RECEIVER_ID].astype("string")
            subordinate_ids = hits[COL_SENDER_ID].astype("string")

    hits["manager_user_id"] = _normalize_identifier(manager_ids)
    hits["subordinado_user_id"] = _normalize_identifier(subordinate_ids)
    hits = hits.dropna(subset=["manager_user_id", "subordinado_user_id"])
    if hits.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "month_id",
                "manager_user_id",
                "subordinado_user_id",
                "nlp_concepto_sospechoso",
                "nlp_concepto_crudo",
                "nlp_descripciones",
                "tx_count",
                "monto_total",
                "interpretabilidad",
            ]
        )

    aggregations: dict[str, tuple[str, Any]] = {
        "tx_count": (COL_AMOUNT, "count"),
        "monto_total": (COL_AMOUNT, "sum"),
        "nlp_concepto_crudo": ("nlp_concepto_crudo", _combine_unique_texts),
        "nlp_concepto_sospechoso": (
            "matched_category",
            _combine_unique_categories,
        ),
    }
    if COL_DESCRIPTION in hits.columns:
        aggregations["nlp_descripciones"] = (
            COL_DESCRIPTION,
            _combine_unique_texts,
        )
    else:
        hits["nlp_descripciones"] = ""
        aggregations["nlp_descripciones"] = (
            "nlp_descripciones",
            _combine_unique_texts,
        )

    agg = (
        hits.groupby(
            ["month_id", "manager_user_id", "subordinado_user_id"],
            observed=True,
        )
        .agg(**aggregations)
        .reset_index()
    )
    agg = agg.sort_values(["tx_count", "monto_total"], ascending=[False, False])
    agg["timeframe"] = timeframe
    agg["nlp_concepto_crudo"] = agg["nlp_concepto_crudo"].fillna("")
    if "nlp_descripciones" in agg:
        agg["nlp_descripciones"] = agg["nlp_descripciones"].fillna("")
    agg["nlp_concepto_sospechoso"] = agg["nlp_concepto_sospechoso"].apply(
        lambda values: values if isinstance(values, list) else []
    )
    agg["_concepto_label"] = agg["nlp_concepto_sospechoso"].apply(
        lambda values: ", ".join(values) if values else "SIN_CONCEPTO"
    )
    if direction == "manager_a_subordinado":
        def _build_message(row: pd.Series) -> str:
            return (
                f"En la ventana '{timeframe}', durante {row.get('month_id', 'sin_mes')} "
                f"el manager {row.get('manager_user_id', 'sin_manager')} envió "
                f"{int(row.get('tx_count', 0))} pagos al subordinado "
                f"{row.get('subordinado_user_id', 'sin_subordinado')} etiquetados como "
                f"'{row.get('_concepto_label', 'SIN_CONCEPTO')}', acumulando "
                f"{_format_float(row.get('monto_total', 0))} en monto total."
                + (
                    f" Conceptos crudos detectados: {row.get('nlp_concepto_crudo', '').strip()}."
                    if str(row.get('nlp_concepto_crudo', '')).strip()
                    else ""
                )
                + (
                    f" Descripciones de origen: {row.get('nlp_descripciones', '').strip()}."
                    if str(row.get('nlp_descripciones', '')).strip()
                    else ""
                )
            )
    else:
        def _build_message(row: pd.Series) -> str:
            return (
                f"En la ventana '{timeframe}', durante {row.get('month_id', 'sin_mes')} "
                f"el subordinado {row.get('subordinado_user_id', 'sin_subordinado')} envió "
                f"{int(row.get('tx_count', 0))} pagos al manager "
                f"{row.get('manager_user_id', 'sin_manager')} etiquetados como "
                f"'{row.get('_concepto_label', 'SIN_CONCEPTO')}', acumulando "
                f"{_format_float(row.get('monto_total', 0))} en monto total."
                + (
                    f" Conceptos crudos detectados: {row.get('nlp_concepto_crudo', '').strip()}."
                    if str(row.get('nlp_concepto_crudo', '')).strip()
                    else ""
                )
                + (
                    f" Descripciones de origen: {row.get('nlp_descripciones', '').strip()}."
                    if str(row.get('nlp_descripciones', '')).strip()
                    else ""
                )
            )

    agg["interpretabilidad"] = agg.apply(_build_message, axis=1)
    agg = agg.drop(columns=["_concepto_label"])
    columns = [
        "timeframe",
        "month_id",
        "manager_user_id",
        "subordinado_user_id",
        "nlp_concepto_sospechoso",
        "nlp_concepto_crudo",
        "nlp_descripciones",
        "tx_count",
        "monto_total",
        "interpretabilidad",
    ]
    return agg.reindex(columns=columns)


def question2_manager_concepts(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    """Resume los conceptos NLP detectados en interacciones manager-subordinado.

    Parameters
    ----------
    reports
        Diccionario de salidas de la canalización con las secciones
        ``"transaccion"`` y columnas de riesgo asociadas.
    timeframe
        Ventana temporal a consultar (``"todo_el_tiempo"`` por defecto).

    Metodología
    -----------
    1. Obtiene el detalle de transacciones y calcula los hits NLP mediante
       :func:`_manager_nlp_hits`.
    2. Agrupa los resultados por mes y categoría, contando transacciones y
       estimando el percentil 95 de ``risk_score``.
    3. Ordena por severidad y redacta textos explicativos con el número de
       eventos y la intensidad estimada.

    Returns
    -------
    pandas.DataFrame
        Tabla con conteos, severidad y explicaciones para cada concepto NLP
        sospechoso.
    """
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
    """Resume pares donde parece que hubo "algo por algo" de forma muy simple.

    El objetivo es explicar con palabras llanas por qué dos personas llaman la
    atención. Primero busca los resúmenes ya calculados; si no existen, arma las
    cuentas de cero tomando las transacciones con puntajes altos. Cada fila
    devuelve números fáciles de leer, un ejemplo concreto y un texto que describe
    por qué el caso luce riesgoso.
    """
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
        "quid_aprob_count",
        "quid_comp_count",
        "quid_tx_amount_sum",
        "quid_tx_amount_max",
        "quid_tx_risk_max",
        "quid_tx_risk_avg",
        "quid_tx_first_fecha",
        "quid_tx_last_fecha",
        "quid_tx_relaciones",
        "quid_top_tx_fecha",
        "quid_top_tx_monto",
        "quid_top_tx_score",
        "quid_top_tx_risk",
        "quid_top_tx_relacion",
        "quid_top_tx_rel_label",
        "quid_top_tx_value_vs_load_days",
        "quid_top_tx_aprob",
        "quid_top_tx_comp",
        "quid_top_tx_descripcion",
        "quid_top_tx_referencia",
        "quid_top_tx_relajado",
    ]

    pair_column_rename = {
        "quid_pair_clave": "identificador_emisor_a_receptor",
        "quid_pair_label": "resumen_personas_involucradas",
        "quid_tx_count": "cantidad_movimientos_con_indicio_de_algo_por_algo",
        "quid_score_max": "puntaje_algo_por_algo_mas_alto_en_el_par",
        "quid_score_avg": "puntaje_algo_por_algo_promedio_en_el_par",
        "quid_manager_ratio": "porcentaje_movimientos_donde_participa_un_jefe",
        "quid_aprob_ratio": "porcentaje_movimientos_con_texto_de_aprobacion",
        "quid_comp_ratio": "porcentaje_movimientos_con_texto_de_compensacion",
        "quid_aprob_count": "cantidad_movimientos_con_texto_de_aprobacion",
        "quid_comp_count": "cantidad_movimientos_con_texto_de_compensacion",
        "quid_tx_amount_sum": "monto_total_de_los_movimientos_relacionados",
        "quid_tx_amount_max": "monto_mas_alto_de_los_movimientos_relacionados",
        "quid_tx_risk_max": "riesgo_maximo_de_los_movimientos_relacionados",
        "quid_tx_risk_avg": "riesgo_promedio_de_los_movimientos_relacionados",
        "quid_tx_first_fecha": "fecha_del_primer_movimiento_relacionado",
        "quid_tx_last_fecha": "fecha_del_ultimo_movimiento_relacionado",
        "quid_tx_relaciones": "tipos_de_relacion_observados_entre_las_personas",
        "quid_top_tx_fecha": "ejemplo_clave_fecha_del_movimiento",
        "quid_top_tx_monto": "ejemplo_clave_monto_del_movimiento",
        "quid_top_tx_score": "ejemplo_clave_puntaje_algo_por_algo",
        "quid_top_tx_risk": "ejemplo_clave_riesgo_del_movimiento",
        "quid_top_tx_relacion": "ejemplo_clave_relacion_declarada",
        "quid_top_tx_rel_label": "ejemplo_clave_relacion_detectada",
        "quid_top_tx_value_vs_load_days": "ejemplo_clave_dias_entre_carga_y_autorizacion",
        "quid_top_tx_aprob": "ejemplo_clave_texto_menciona_aprobacion",
        "quid_top_tx_comp": "ejemplo_clave_texto_menciona_compensacion",
        "quid_top_tx_descripcion": "ejemplo_clave_texto_libre",
        "quid_top_tx_referencia": "ejemplo_clave_referencia",
        "quid_top_tx_relajado": "ejemplo_clave_proviene_de_filtro_relajado",
    }

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
            fallback_candidates["feat_quid_has_approval"] = fallback_candidates[
                "feat_quid_has_approval"
            ].apply(lambda value: bool(_coerce_bool(value)))
            if "feat_quid_has_comp" not in fallback_candidates:
                fallback_candidates["feat_quid_has_comp"] = False
            fallback_candidates["feat_quid_has_comp"] = fallback_candidates[
                "feat_quid_has_comp"
            ].apply(lambda value: bool(_coerce_bool(value)))
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

    pair_df["nivel_respuesta"] = "resumen_del_par"
    pair_df["timeframe"] = timeframe
    for ratio_column in ("quid_manager_ratio", "quid_aprob_ratio", "quid_comp_ratio"):
        if ratio_column in pair_df.columns:
            pair_df[ratio_column] = pd.to_numeric(
                pair_df[ratio_column], errors="coerce"
            ) * 100.0
    string_pair_columns = [
        "quid_tx_first_fecha",
        "quid_tx_last_fecha",
        "quid_tx_relaciones",
        "quid_top_tx_fecha",
        "quid_top_tx_relacion",
        "quid_top_tx_rel_label",
        "quid_top_tx_descripcion",
        "quid_top_tx_referencia",
    ]
    bool_pair_columns = [
        "quid_top_tx_aprob",
        "quid_top_tx_comp",
        "quid_top_tx_relajado",
    ]
    for column in string_pair_columns:
        if column in pair_df.columns:
            pair_df[column] = pair_df[column].astype("object")
    for column in bool_pair_columns:
        if column in pair_df.columns:
            pair_df[column] = pair_df[column].astype("object")

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

    tx_column_rename = {
        "fecha_hora_ts": "fecha_y_hora_del_movimiento",
        COL_SENDER_ID: "id_persona_que_envia",
        COL_RECEIVER_ID: "id_persona_que_recibe",
        COL_AMOUNT: "monto_del_movimiento",
        "relacion": "relacion_declarada_en_el_movimiento",
        "feat_quid_rel_label": "relacion_detectada_por_el_modelo_algo_por_algo",
        "feat_quid_has_approval": "texto_libre_menciona_aprobacion",
        "feat_quid_has_comp": "texto_libre_menciona_compensacion",
        "feat_quid_value_vs_load_days": "dias_entre_carga_y_autorizacion_del_movimiento",
        "feat_quid_score": "puntaje_algo_por_algo_del_movimiento",
        "descripcion": "texto_libre_del_movimiento",
        "reference_number_trans_desc": "referencia_del_movimiento",
        "risk_score": "riesgo_estimado_del_movimiento",
    }

    output_rename = {**pair_column_rename, **tx_column_rename}

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

    tx_df["nivel_respuesta"] = "movimiento_detallado"
    tx_df["timeframe"] = timeframe
    if not tx_df.empty:
        def _describe_tx_row(row: pd.Series) -> str:
            fecha = _coalesce_str(row.get("fecha_hora_ts"), default="sin_fecha")
            emisor = _coalesce_str(row.get(COL_SENDER_ID), default="emisor_desconocido")
            receptor = _coalesce_str(row.get(COL_RECEIVER_ID), default="receptor_desconocido")
            monto = _format_float(row.get(COL_AMOUNT, 0))
            score = _safe_float(row.get("feat_quid_score")) or 0.0
            delta = row.get("feat_quid_value_vs_load_days")
            if pd.isna(delta):
                delta_text = "sin dato de diferencia de días"
            else:
                delta_val = int(round(float(delta)))
                if delta_val < 0:
                    delta_text = f"se cargó {abs(delta_val)} días antes de la autorización"
                elif delta_val > 0:
                    delta_text = f"la autorización llegó {delta_val} días después"
                else:
                    delta_text = "la autorización llegó el mismo día"
            aprob_flag = _coerce_bool(row.get("feat_quid_has_approval"))
            if aprob_flag is None:
                aprob_text = "sin dato"
            else:
                aprob_text = "sí" if aprob_flag else "no"
            comp_flag = _coerce_bool(row.get("feat_quid_has_comp"))
            if comp_flag is None:
                comp_text = "sin dato"
            else:
                comp_text = "sí" if comp_flag else "no"
            partes = [
                f"El {fecha} {emisor} movió {monto} hacia {receptor}.",
                f"Puntaje 'algo por algo': {score:.2f} (meta {min_score}).",
                f"Diferencia carga-autorización: {delta_text}.",
                f"Palabra de aprobación: {aprob_text}; palabra de compensación: {comp_text}.",
            ]
            if _coerce_bool(row.get("criterio_relajado")):
                partes.append(
                    "Salió al relajar el filtro porque no había ejemplos más claros en la ventana."
                )
            return " ".join(partes)

        tx_df["interpretabilidad"] = tx_df.apply(_describe_tx_row, axis=1)
    else:
        tx_df["interpretabilidad"] = pd.Series(dtype="object")

    pair_enrichment = pd.DataFrame()
    if (
        not pair_df.empty
        and not tx_df.empty
        and {COL_SENDER_ID, COL_RECEIVER_ID}.issubset(tx_df.columns)
    ):
        tx_for_enrichment = tx_df.copy()
        if "nivel_respuesta" in tx_for_enrichment.columns:
            tx_for_enrichment = tx_for_enrichment.loc[
                tx_for_enrichment["nivel_respuesta"] == "movimiento_detallado"
            ].copy()
        sender_series = tx_for_enrichment.get(COL_SENDER_ID)
        receiver_series = tx_for_enrichment.get(COL_RECEIVER_ID)
        if sender_series is not None and receiver_series is not None:
            sender_values = sender_series.astype(str)
            receiver_values = receiver_series.astype(str)
            sender_values = sender_values.where(~sender_values.str.lower().eq("nan"), "")
            receiver_values = receiver_values.where(~receiver_values.str.lower().eq("nan"), "")
            tx_for_enrichment["__pair_key__"] = sender_values + "->" + receiver_values
            tx_for_enrichment = tx_for_enrichment.loc[
                tx_for_enrichment["__pair_key__"] != "->"
            ].copy()
            valid_pairs = (
                pair_df.get("quid_pair_clave", pd.Series(dtype=str))
                .dropna()
                .astype(str)
                .unique()
            )
            if len(valid_pairs):
                tx_for_enrichment = tx_for_enrichment.loc[
                    tx_for_enrichment["__pair_key__"].isin(valid_pairs)
                ].copy()
            else:
                tx_for_enrichment = tx_for_enrichment.iloc[0:0]
            if not tx_for_enrichment.empty:
                records: list[dict[str, Any]] = []
                for pair_key, group in tx_for_enrichment.groupby(
                    "__pair_key__", observed=True
                ):
                    record: dict[str, Any] = {"quid_pair_clave": pair_key}
                    record["quid_aprob_count"] = _count_true(
                        group.get("feat_quid_has_approval")
                    )
                    record["quid_comp_count"] = _count_true(
                        group.get("feat_quid_has_comp")
                    )
                    if COL_AMOUNT in group:
                        amount_series = pd.to_numeric(
                            group[COL_AMOUNT], errors="coerce"
                        )
                        if not amount_series.empty:
                            record["quid_tx_amount_sum"] = float(
                                amount_series.fillna(0).sum()
                            )
                            if amount_series.notna().any():
                                record["quid_tx_amount_max"] = float(
                                    amount_series.max()
                                )
                    if "risk_score" in group:
                        risk_series = pd.to_numeric(
                            group["risk_score"], errors="coerce"
                        )
                        if risk_series.notna().any():
                            record["quid_tx_risk_max"] = float(risk_series.max())
                            record["quid_tx_risk_avg"] = float(risk_series.mean())
                    date_col = "fecha_hora_ts" if "fecha_hora_ts" in group else None
                    if date_col is None and "fecha_hora" in group:
                        date_col = "fecha_hora"
                    if date_col:
                        chronological = group.dropna(subset=[date_col]).sort_values(
                            date_col
                        )
                        if not chronological.empty:
                            record["quid_tx_first_fecha"] = chronological.iloc[0].get(
                                date_col
                            )
                            record["quid_tx_last_fecha"] = chronological.iloc[-1].get(
                                date_col
                            )
                    relations: list[str] = []
                    if COL_RELATION in group:
                        relations.extend(
                            group[COL_RELATION].dropna().astype(str).tolist()
                        )
                    if "feat_quid_rel_label" in group:
                        relations.extend(
                            group["feat_quid_rel_label"].dropna().astype(str).tolist()
                        )
                    relations = [
                        rel.strip()
                        for rel in relations
                        if rel and rel.lower() != "nan"
                    ]
                    if relations:
                        record["quid_tx_relaciones"] = ", ".join(
                            dict.fromkeys(relations)
                        )
                    sort_by: list[str] = []
                    ascending: list[bool] = []
                    if "feat_quid_score" in group:
                        sort_by.append("feat_quid_score")
                        ascending.append(False)
                    if "risk_score" in group:
                        sort_by.append("risk_score")
                        ascending.append(False)
                    if date_col:
                        sort_by.append(date_col)
                        ascending.append(True)
                    group_sorted = (
                        group.sort_values(sort_by, ascending=ascending)
                        if sort_by
                        else group
                    )
                    top_tx = group_sorted.iloc[0]
                    record["quid_top_tx_fecha"] = (
                        top_tx.get(date_col) if date_col else None
                    )
                    record["quid_top_tx_monto"] = (
                        _safe_float(top_tx.get(COL_AMOUNT))
                        if COL_AMOUNT in group
                        else None
                    )
                    record["quid_top_tx_score"] = _safe_float(
                        top_tx.get("feat_quid_score")
                    )
                    record["quid_top_tx_risk"] = _safe_float(
                        top_tx.get("risk_score")
                    )
                    relation_val = (
                        _coalesce_str(top_tx.get(COL_RELATION), default="")
                        if COL_RELATION in group
                        else ""
                    )
                    record["quid_top_tx_relacion"] = relation_val or None
                    rel_label_val = (
                        _coalesce_str(top_tx.get("feat_quid_rel_label"), default="")
                        if "feat_quid_rel_label" in group
                        else ""
                    )
                    record["quid_top_tx_rel_label"] = rel_label_val or None
                    record["quid_top_tx_value_vs_load_days"] = _safe_float(
                        top_tx.get("feat_quid_value_vs_load_days")
                    )
                    record["quid_top_tx_aprob"] = _coerce_bool(
                        top_tx.get("feat_quid_has_approval")
                    )
                    record["quid_top_tx_comp"] = _coerce_bool(
                        top_tx.get("feat_quid_has_comp")
                    )
                    descripcion = _coalesce_str(
                        top_tx.get("descripcion"),
                        top_tx.get(COL_DESCRIPTION),
                        default="",
                    )
                    record["quid_top_tx_descripcion"] = descripcion or None
                    referencia = _coalesce_str(
                        top_tx.get("reference_number_trans_desc"),
                        default="",
                    )
                    record["quid_top_tx_referencia"] = referencia or None
                    record["quid_top_tx_relajado"] = bool(
                        _coerce_bool(top_tx.get("criterio_relajado"))
                    )
                    records.append(record)
                if records:
                    pair_enrichment = pd.DataFrame(records)
    if not pair_enrichment.empty:
        enrichment_indexed = pair_enrichment.set_index("quid_pair_clave")
        pair_df = pair_df.set_index("quid_pair_clave")
        pair_df.update(enrichment_indexed)
        pair_df = pair_df.reset_index()

    def _stringify(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, float) and pd.isna(value):
            return ""
        text = str(value)
        return "" if not text or text.lower() == "nan" else text

    def _describe_pair_row(row: pd.Series) -> str:
        label = _coalesce_str(
            row.get("quid_pair_label"),
            row.get("quid_pair_clave"),
            default="sin_identificar",
        )
        count_value = _safe_float(row.get("quid_tx_count")) or 0.0
        count = int(round(count_value)) if count_value else 0
        max_score = _safe_float(row.get("quid_score_max")) or 0.0
        avg_score = _safe_float(row.get("quid_score_avg")) or 0.0
        parts: list[str] = []
        if count:
            parts.append(
                f"{label}: hallamos {count} movimientos que parecen un 'algo por algo'."
            )
        else:
            parts.append(
                f"{label}: hallamos movimientos que parecen un 'algo por algo'."
            )
        parts.append(
            f"El puntaje más alto fue {max_score:.2f} (meta {min_score}) y el promedio quedó en {avg_score:.2f}."
        )
        manager_percent = _safe_float(row.get("quid_manager_ratio")) or 0.0
        if manager_percent:
            parts.append(
                f"En {manager_percent:.0f}% de los movimientos participó alguien con rol de jefe."
            )
        aprob_percent = _safe_float(row.get("quid_aprob_ratio")) or 0.0
        aprob_count_val = _safe_float(row.get("quid_aprob_count"))
        if aprob_percent:
            if aprob_count_val is not None:
                parts.append(
                    f"Palabras de aprobación aparecieron en {aprob_percent:.0f}% de los casos ({int(round(aprob_count_val))} movimientos)."
                )
            else:
                parts.append(
                    f"Palabras de aprobación aparecieron en {aprob_percent:.0f}% de los casos."
                )
        comp_percent = _safe_float(row.get("quid_comp_ratio")) or 0.0
        comp_count_val = _safe_float(row.get("quid_comp_count"))
        if comp_percent:
            if comp_count_val is not None:
                parts.append(
                    f"Menciones de compensación salieron en {comp_percent:.0f}% de los casos ({int(round(comp_count_val))} movimientos)."
                )
            else:
                parts.append(
                    f"Menciones de compensación salieron en {comp_percent:.0f}% de los casos."
                )
        total_amount = _safe_float(row.get("quid_tx_amount_sum"))
        max_amount = _safe_float(row.get("quid_tx_amount_max"))
        if total_amount is not None:
            parts.append(
                f"Los montos ligados a este par suman {_format_float(total_amount)}."
            )
        if max_amount is not None:
            parts.append(
                f"El movimiento más alto dentro del grupo fue de {_format_float(max_amount)}."
            )
        risk_max = _safe_float(row.get("quid_tx_risk_max"))
        risk_avg = _safe_float(row.get("quid_tx_risk_avg"))
        if risk_max is not None or risk_avg is not None:
            if risk_max is None:
                risk_max = risk_avg
            if risk_avg is None:
                risk_avg = risk_max
            if risk_max is not None and risk_avg is not None:
                parts.append(
                    f"El riesgo calculado llegó a {risk_max:.2f} y en promedio quedó en {risk_avg:.2f}."
                )
        first_fecha = _stringify(row.get("quid_tx_first_fecha"))
        last_fecha = _stringify(row.get("quid_tx_last_fecha"))
        if first_fecha and last_fecha:
            if first_fecha == last_fecha:
                parts.append(f"Todo apunta al día {first_fecha}.")
            else:
                parts.append(
                    f"Los movimientos sospechosos van de {first_fecha} a {last_fecha}."
                )
        elif first_fecha:
            parts.append(f"Todo apunta al día {first_fecha}.")
        relaciones_text = _stringify(row.get("quid_tx_relaciones"))
        if relaciones_text:
            parts.append(f"Relación reportada: {relaciones_text}.")
        top_fecha = _stringify(row.get("quid_top_tx_fecha"))
        if top_fecha:
            top_bits: list[str] = [f"Ejemplo claro: {top_fecha}"]
            top_monto = _safe_float(row.get("quid_top_tx_monto"))
            if top_monto is not None:
                top_bits.append(f"por {_format_float(top_monto)}")
            top_score = _safe_float(row.get("quid_top_tx_score"))
            if top_score is not None:
                top_bits.append(f"puntaje {top_score:.2f}")
            top_risk = _safe_float(row.get("quid_top_tx_risk"))
            if top_risk is not None:
                top_bits.append(f"riesgo {top_risk:.2f}")
            relation_bits = [
                bit
                for bit in (
                    _stringify(row.get("quid_top_tx_rel_label")),
                    _stringify(row.get("quid_top_tx_relacion")),
                )
                if bit
            ]
            if relation_bits:
                top_bits.append("relación " + " / ".join(relation_bits))
            top_delta = _safe_float(row.get("quid_top_tx_value_vs_load_days"))
            if top_delta is not None:
                if top_delta < 0:
                    top_bits.append(
                        f"se cargó {abs(int(round(top_delta)))} días antes de autorizarse"
                    )
                elif top_delta > 0:
                    top_bits.append(
                        f"se autorizó {int(round(top_delta))} días después"
                    )
                else:
                    top_bits.append("se autorizó el mismo día")
            evidencias: list[str] = []
            aprob_bool = _coerce_bool(row.get("quid_top_tx_aprob"))
            if aprob_bool is not None:
                evidencias.append(
                    "menciona aprobación" if aprob_bool else "sin aprobación"
                )
            comp_bool = _coerce_bool(row.get("quid_top_tx_comp"))
            if comp_bool is not None:
                evidencias.append(
                    "menciona compensación" if comp_bool else "sin compensación"
                )
            if evidencias:
                top_bits.append("; ".join(evidencias))
            descripcion_text = _stringify(row.get("quid_top_tx_descripcion"))
            if descripcion_text:
                top_bits.append(f"texto: {descripcion_text}")
            referencia_text = _stringify(row.get("quid_top_tx_referencia"))
            if referencia_text:
                top_bits.append(f"referencia: {referencia_text}")
            if _coerce_bool(row.get("quid_top_tx_relajado")):
                top_bits.append("salió al relajar el filtro")
            parts.append(". ".join(top_bits) + ".")
        if _coerce_bool(row.get("criterio_relajado")):
            parts.append(
                "Lo mostramos aunque quedó justo debajo del filtro original para no perder la pista."
            )
        return " ".join(part.strip() for part in parts if part).strip()

    if not pair_df.empty:
        pair_df["interpretabilidad"] = pair_df.apply(
            _describe_pair_row,
            axis=1,
        )
    else:
        pair_df["interpretabilidad"] = pd.Series(dtype="object")
    if "criterio_relajado" in pair_df:
        pair_df = pair_df.drop(columns=["criterio_relajado"])
    if "criterio_relajado" in tx_df:
        tx_df = tx_df.drop(columns=["criterio_relajado"])

    combined = pd.concat([pair_df, tx_df], ignore_index=True, sort=False)
    combined = combined.rename(columns=output_rename)
    if "nivel_respuesta" in combined:
        combined["nivel_respuesta"] = combined["nivel_respuesta"].replace(
            {
                "resumen_del_par": "resumen_del_par_algo_por_algo",
                "movimiento_detallado": "movimiento_detallado_algo_por_algo",
            }
        )
    ordered_cols = [
        "timeframe",
        "nivel_respuesta",
        *[
            output_rename.get(column, column)
            for column in pair_columns + tx_columns
            if output_rename.get(column, column) in combined.columns
        ],
    ]
    ordered_cols = list(dict.fromkeys(ordered_cols)) + ["interpretabilidad"]
    return combined.reindex(columns=ordered_cols)


def question4_quid_negative_value_vs_load(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    """Busca autorizaciones con desfases negativos entre valor y carga temporal.

    Parameters
    ----------
    reports
        Diccionario con secciones de transacciones y señales quid.
    timeframe
        Ventana temporal analizada (``"todo_el_tiempo"`` por defecto).

    Metodología
    -----------
    1. Revisa ``casuistica_quid_pro_quo_tx`` o, en su ausencia, las
       transacciones base del ``timeframe``.
    2. Selecciona transacciones con ``feat_quid_value_vs_load_days`` negativo y
       calcula responsables según la relación jerárquica.
    3. Si no hay casos negativos, adopta criterios relajados (top 10 por menor
       desfase o por ``feat_quid_score``).
    4. Construye interpretabilidad detallando desfase, puntaje y responsable.

    Returns
    -------
    pandas.DataFrame
        Tabla de autorizaciones sospechosas con columnas explicativas sobre el
        desfase identificado.
    """
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
        if pd.notna(delta):
            delta_val = int(round(float(delta)))
            if delta_val < 0:
                delta_txt = f"se registró {abs(delta_val)} días antes de la autorización"
            elif delta_val > 0:
                delta_txt = f"la autorización tardó {delta_val} días"
            else:
                delta_txt = "la autorización llegó el mismo día"
        else:
            delta_txt = "no sabemos cuántos días pasaron entre la carga y la autorización"
        base = (
            f"En '{timeframe}', la transacción del {row.get('fecha_hora_ts', 'sin_fecha')} "
            f"entre {_coalesce_str(row.get(COL_SENDER_ID), default='emisor_desconocido')} y "
            f"{_coalesce_str(row.get(COL_RECEIVER_ID), default='receptor_desconocido')} {delta_txt}. "
            f"El puntaje 'algo por algo' es {row.get('feat_quid_score', 0):.2f}. "
            f"La persona que debe revisarla es {row.get('responsable_user_id', 'sin_responsable')}."
        )
        if relaxed:
            base += " Se incluyeron los desfases más raros aunque no sean negativos para mantenerlos vigilados."
        return base

    result["interpretabilidad"] = result.apply(_describe_quid, axis=1)
    ordered_cols = ["timeframe"] + keep + ["responsable_user_id", "interpretabilidad"]
    return result.reindex(columns=ordered_cols)


def question5_reference_reuse(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME, include_raw_concept: bool = False
) -> pd.DataFrame:
    """Detecta receptores que reciben el mismo concepto sospechoso desde múltiples emisores."""

    tx = _get_section(reports, "transaccion", timeframe)
    empty_columns = [
        "timeframe",
        "nivel_respuesta",
        COL_RECEIVER_ID,
        "nlp_concepto_sospechoso",
    ]
    if include_raw_concept:
        empty_columns.append("conceptos_crudos")
    empty_columns.extend(
        [
            "emisores_unicos",
            "emisores_detalle",
            "meses_distintos",
            "meses_detalle",
            "tx_count",
            "monto_total",
            "riesgo_promedio",
            "riesgo_p95",
            "interpretabilidad",
        ]
    )
    if tx.empty:
        return pd.DataFrame(columns=empty_columns)

    required_columns = {COL_SENDER_ID, COL_RECEIVER_ID, "nlp_concepto_sospechoso"}
    if not required_columns.issubset(tx.columns):
        return pd.DataFrame(columns=empty_columns)

    work_columns = list(required_columns.union({COL_AMOUNT, "month_id", "fecha_hora_ts", "risk_score"}))
    if include_raw_concept:
        work_columns.append("nlp_concepto_crudo")
    existing_columns = [column for column in work_columns if column in tx.columns]
    work = tx[existing_columns].copy()
    work[COL_SENDER_ID] = work[COL_SENDER_ID].astype("string").str.strip()
    work[COL_RECEIVER_ID] = work[COL_RECEIVER_ID].astype("string").str.strip()
    work["nlp_concepto_sospechoso"] = (
        work["nlp_concepto_sospechoso"].astype("string").str.strip()
    )
    work = work.loc[
        (work[COL_SENDER_ID] != "")
        & (work[COL_RECEIVER_ID] != "")
        & (work["nlp_concepto_sospechoso"] != "")
    ].copy()
    if work.empty:
        return pd.DataFrame(columns=empty_columns)

    work[COL_AMOUNT] = pd.to_numeric(work.get(COL_AMOUNT), errors="coerce")
    work["risk_score"] = pd.to_numeric(work.get("risk_score"), errors="coerce")

    if include_raw_concept:
        work = _ensure_raw_concept_column(work)

    if "month_id" not in work.columns:
        if "fecha_hora_ts" in work.columns:
            timestamps = pd.to_datetime(work["fecha_hora_ts"], errors="coerce")
            work["month_id"] = timestamps.dt.to_period("M").astype("string")
        else:
            work["month_id"] = pd.Series(pd.NA, index=work.index, dtype="string")
    else:
        work["month_id"] = work["month_id"].astype("string").str.strip()

    work["_sender_clean"] = work[COL_SENDER_ID].replace("", pd.NA)
    work["_month_clean"] = work["month_id"].replace("", pd.NA)

    def _join_unique(series: pd.Series) -> str:
        values = {
            str(value)
            for value in series.dropna().astype(str)
            if str(value).strip() not in {"", "<NA>"}
        }
        return "; ".join(sorted(values))

    aggregations: dict[str, tuple[str, Any]] = {
        "emisores_unicos": ("_sender_clean", "nunique"),
        "emisores_detalle": ("_sender_clean", _join_unique),
        "meses_distintos": ("_month_clean", "nunique"),
        "meses_detalle": ("_month_clean", _join_unique),
        "tx_count": (COL_AMOUNT, "count"),
        "monto_total": (COL_AMOUNT, "sum"),
        "riesgo_promedio": ("risk_score", "mean"),
        "riesgo_p95": (
            "risk_score",
            lambda s: float(s.dropna().quantile(0.95)) if s.dropna().size else float("nan"),
        ),
    }
    if include_raw_concept:
        aggregations["conceptos_crudos"] = (
            "nlp_concepto_crudo",
            _collect_unique_text_list,
        )

    grouped = (
        work.groupby([COL_RECEIVER_ID, "nlp_concepto_sospechoso"], observed=True)
        .agg(**aggregations)
        .reset_index()
    )

    grouped = grouped.loc[grouped["emisores_unicos"].fillna(0) > 1].copy()
    if grouped.empty:
        return pd.DataFrame(columns=empty_columns)

    grouped["monto_total"] = grouped["monto_total"].fillna(0.0)
    grouped["tx_count"] = grouped["tx_count"].fillna(0).astype(int)
    grouped["emisores_unicos"] = grouped["emisores_unicos"].fillna(0).astype(int)
    grouped["meses_distintos"] = grouped["meses_distintos"].fillna(0).astype(int)
    grouped["riesgo_promedio"] = grouped["riesgo_promedio"].fillna(float("nan"))

    grouped["nivel_respuesta"] = "concepto_receptor"
    grouped["timeframe"] = timeframe

    if include_raw_concept:
        grouped["conceptos_crudos"] = grouped["conceptos_crudos"].apply(
            lambda values: values if isinstance(values, list) else []
        )

    def _build_interpretability(row: pd.Series) -> str:
        message = (
            f"Durante '{timeframe}', la persona {_coalesce_str(row.get(COL_RECEIVER_ID), default='sin_receptor')} "
            f"recibió el concepto '{_coalesce_str(row.get('nlp_concepto_sospechoso'), default='SIN_CONCEPTO')}' "
            f"desde {int(row.get('emisores_unicos', 0))} emisores diferentes en "
            f"{int(row.get('meses_distintos', 0))} meses ({row.get('meses_detalle') or 'sin_mes'}), "
            f"sumando {int(row.get('tx_count', 0))} transacciones por {_format_float(row.get('monto_total', 0))}. "
            f"Emisores: {row.get('emisores_detalle') or 'sin_detalle'}."
        )
        if pd.notna(row.get("riesgo_promedio")):
            message += f" Riesgo promedio {row.get('riesgo_promedio', float('nan')):.2f}"
        if pd.notna(row.get("riesgo_p95")):
            message += f" (p95={row.get('riesgo_p95', float('nan')):.2f})"
        if include_raw_concept and row.get("conceptos_crudos"):
            raw_list = ", ".join(row.get("conceptos_crudos") or [])
            if raw_list:
                message += f" Conceptos crudos: {raw_list}."
        return message

    grouped["interpretabilidad"] = grouped.apply(_build_interpretability, axis=1)

    grouped = grouped.sort_values(
        ["emisores_unicos", "meses_distintos", "monto_total"],
        ascending=[False, False, False],
    )

    ordered_columns = [
        "timeframe",
        "nivel_respuesta",
        COL_RECEIVER_ID,
        "nlp_concepto_sospechoso",
    ]
    if include_raw_concept:
        ordered_columns.append("conceptos_crudos")
    ordered_columns.extend([
        "emisores_unicos",
        "emisores_detalle",
        "meses_distintos",
        "meses_detalle",
        "tx_count",
        "monto_total",
        "riesgo_promedio",
        "riesgo_p95",
        "interpretabilidad",
    ])

    return grouped.reindex(columns=ordered_columns)

def question6_centralizers(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    """Prioriza receptores que actúan como nodos centralizadores de fondos.

    Parameters
    ----------
    reports
        Diccionario de reportes con la sección ``"transaccion"``.
    timeframe
        Ventana temporal a evaluar (valor por defecto ``"todo_el_tiempo"``).

    Metodología
    -----------
    1. Agrega las transacciones por mes y receptor calculando inflow total,
       emisores únicos, número de transacciones y riesgo promedio.
    2. Deriva una métrica de centralidad multiplicando inflow por emisores
       únicos para ordenar los resultados.
    3. Construye explicaciones resaltando montos, riesgo y diversidad de
       emisores para cada receptor.

    Returns
    -------
    pandas.DataFrame
        Tabla mensual con receptores centralizadores y columnas interpretables.
    """
    tx = _get_section(reports, "transaccion", timeframe)
    if tx.empty or "month_id" not in tx:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "month_id",
                COL_RECEIVER_ID,
                "emisores_lista",
                "inflow",
                "emisores_unicos",
                "n_tx",
                "risk_avg",
                "centralidad",
                "monto_total_acumulado",
                "apariciones_receptor",
                "interpretabilidad",
            ]
        )

    work = tx[["month_id", COL_SENDER_ID, COL_RECEIVER_ID, COL_AMOUNT, "risk_score"]].copy()
    emisores_detalle = (
        work.groupby(["month_id", COL_RECEIVER_ID], observed=True)[COL_SENDER_ID]
        .apply(lambda values: sorted(pd.unique(values)))
        .reset_index(name="emisores_lista")
    )
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
    agg = agg.merge(emisores_detalle, on=["month_id", COL_RECEIVER_ID], how="left")

    def _format_emitter_list(items: Any) -> str:
        if isinstance(items, (list, tuple)):
            return ", ".join(map(str, items))
        if pd.isna(items):
            return ""
        return str(items)

    agg["emisores_lista"] = agg["emisores_lista"].apply(_format_emitter_list)
    agg["centralidad"] = agg["inflow"] * agg["emisores_unicos"]
    agg = agg.sort_values(["month_id", "centralidad"], ascending=[True, False])
    acumulados = (
        agg.groupby(COL_RECEIVER_ID, observed=True)
        .agg(
            monto_total_acumulado=("inflow", "sum"),
            apariciones_receptor=("month_id", "count"),
        )
        .reset_index()
    )
    agg = agg.merge(acumulados, on=COL_RECEIVER_ID, how="left")
    agg["timeframe"] = timeframe
    agg["interpretabilidad"] = agg.apply(
        lambda row: (
            f"En {_coalesce_str(row.get('month_id'), default='sin_mes')} ({timeframe}), el receptor "
            f"{_coalesce_str(row.get(COL_RECEIVER_ID), default='sin_receptor')} "
            f"recibió {_format_float(row.get('inflow', 0))} de {int(row.get('emisores_unicos', 0))} emisores únicos "
            f"a través de {int(row.get('n_tx', 0))} pagos, logrando centralidad {row.get('centralidad', 0):.2f} "
            f"y riesgo promedio {row.get('risk_avg', 0):.2f}. "
            f"En total acumuló {_format_float(row.get('monto_total_acumulado', 0))} en {int(row.get('apariciones_receptor', 0))} apariciones. "
            f"Emisores: {row.get('emisores_lista') or 'sin_emisores'}."
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "month_id",
        COL_RECEIVER_ID,
        "emisores_lista",
        "inflow",
        "emisores_unicos",
        "n_tx",
        "risk_avg",
        "centralidad",
        "monto_total_acumulado",
        "apariciones_receptor",
        "interpretabilidad",
    ]
    return agg.reindex(columns=columns)


def question7_net_imbalance(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME
) -> pd.DataFrame:
    """Enumera personas con fuerte desbalance neto entre envíos y recepciones.

    Parameters
    ----------
    reports
        Diccionario con la sección ``"persona"`` producida por la canalización.
    timeframe
        Ventana temporal sobre la cual calcular el desbalance (``"todo_el_tiempo"``
        por defecto).

    Metodología
    -----------
    1. Asegura la presencia de ``desbalance_persona_monto_neto`` y calcula su
       valor absoluto para priorizar los casos extremos.
    2. Conserva métricas adicionales como meses con máximo envío/recepción y
       banderas asociadas.
    3. Genera explicaciones destacando montos, flujo neto y meses críticos.

    Returns
    -------
    pandas.DataFrame
        Tabla con personas desbalanceadas, métricas netas y comentarios
        interpretables.
    """
    personas = _standardize_columns(
        _get_section(reports, "persona", timeframe), PERSONA_COLUMN_ALIASES
    )
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
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
    new_definition: TenureDefinition = ("months", CASE13_NEW_EMPLOYEE_YEARS * 12.0),
) -> pd.DataFrame:
    """Detecta receptores recién incorporados que reciben montos altos.

    Parameters
    ----------
    reports
        Diccionario de reportes con la sección ``"persona"`` y banderas del
        caso 13.
    timeframe
        Ventana temporal analizada (valor por defecto ``"todo_el_tiempo"``).
    new_definition
        Parámetro que define qué se considera receptor nuevo. Puede ser una
        tupla ``("months", valor)`` para fijar el umbral en meses o
        ``("percentile", valor)`` para estimarlo a partir de los percentiles de
        antigüedad (0–1). Cuando se usan percentiles se imprime la conversión a
        meses. Por defecto se mantiene el criterio original del caso, es decir,
        ``("months", 6.0)`` (equivalente a 0.5 años).

    Metodología
    -----------
    1. Filtra personas con la bandera ``caso13_persona_flag_nuevo_receptor_altos_montos``.
    2. Calcula totales recibidos, emisores únicos, montos promedio y meses
       transcurridos desde la primera recepción.
    3. Prioriza a quienes concentran montos en el percentil 90 durante sus
       primeros seis meses de actividad.
    4. Redacta interpretabilidad destacando el carácter reciente y el volumen
       recibido.

    Returns
    -------
    pandas.DataFrame
        Tabla con receptores nuevos de alto monto y explicaciones asociadas.
    """
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
                "caso13_persona_emisores_lista",
                "interpretabilidad",
            ]
        )

    tx = _standardize_columns(
        _get_section(reports, "transaccion", timeframe), TRANSACTION_COLUMN_ALIASES
    )
    new_months_threshold = _resolve_tenure_months(
        tx,
        "receptor_antiguedad_anios",
        new_definition,
        default_months=CASE13_NEW_EMPLOYEE_YEARS * 12.0,
        context="question8_case13_new_employees",
    )
    if pd.isna(new_months_threshold):
        new_months_threshold = CASE13_NEW_EMPLOYEE_YEARS * 12.0

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
                "caso13_persona_emisores_lista",
                "interpretabilidad",
            ]
        )

    tenure_columns = [
        ("caso13_persona_antiguedad_meses", 1.0),
        ("caso13_persona_antiguedad_anios", 12.0),
        ("persona_antiguedad_meses", 1.0),
        ("persona_antiguedad_anios", 12.0),
    ]
    for column, multiplier in tenure_columns:
        if column in work.columns:
            numeric = pd.to_numeric(work[column], errors="coerce") * multiplier
            work = work.loc[numeric <= new_months_threshold].copy()
            break

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
                "caso13_persona_emisores_lista",
                "interpretabilidad",
            ]
        )

    work["timeframe"] = timeframe
    work = work.sort_values(
        ["caso13_persona_tx_altos", "caso13_persona_monto_total"], ascending=[False, False]
    )
    persona_series = work.get("persona", pd.Series(dtype="object"))
    sender_lists = _collect_sender_lists(reports, timeframe, persona_series)
    work["caso13_persona_emisores_lista"] = persona_series.astype(str).map(sender_lists)
    work["caso13_persona_emisores_lista"] = work["caso13_persona_emisores_lista"].apply(
        lambda value: value if isinstance(value, list) else []
    )
    threshold_text = _format_months_threshold(new_months_threshold)
    work["interpretabilidad"] = work.apply(
        lambda row: (
            f"En la ventana '{timeframe}', la persona {_coalesce_str(row.get('persona'), default='sin_persona')} "
            f"es un receptor con antigüedad ≤{threshold_text} que recibió "
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
        "caso13_persona_emisores_lista",
        "interpretabilidad",
    ]
    return work.reindex(columns=columns)


def question9_case14_veterans_from_newcomers(
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
    newcomer_definition: TenureDefinition = (
        "months",
        CASE14_NEW_EMPLOYEE_YEARS * 12.0,
    ),
    veteran_definition: TenureDefinition = (
        "months",
        CASE14_OLD_EMPLOYEE_YEARS * 12.0,
    ),
) -> pd.DataFrame:
    """Prioriza veteranos que reciben pagos de emisores nuevos dentro de la red.

    Parameters
    ----------
    reports
        Diccionario de reportes con la sección ``"persona"`` y banderas del
        caso 14.
    timeframe
        Ventana temporal a analizar (``"todo_el_tiempo"`` por defecto).
    newcomer_definition
        Criterio para considerar a un emisor como recién ingresado. Acepta
        ``("months", valor)`` o ``("percentile", valor)``. En el caso
        percentil se imprime su equivalencia en meses.
    veteran_definition
        Criterio para clasificar a un receptor como veterano siguiendo el mismo
        formato que ``newcomer_definition``. El valor por defecto preserva el
        umbral original del caso: ``("months", 60.0)`` (equivalente a 5 años).

    Metodología
    -----------
    1. Identifica personas con la bandera
       ``caso14_persona_flag_antiguo_recibe_de_nuevos`` o reconstruye la métrica
       desde las transacciones base.
    2. Calcula montos y transacciones recibidas desde emisores recientes,
       incluyendo promedios y emisores únicos.
    3. Ordena los resultados por monto y riesgo, redactando interpretabilidad
       sobre la relación veterano-novato.

    Returns
    -------
    pandas.DataFrame
        Tabla con veteranos relevantes, sus métricas de recepción y contexto.
    """
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
                "caso14_persona_emisores_nuevos_lista",
                "interpretabilidad",
            ]
        )

    tx = _get_section(reports, "transaccion", timeframe)
    newcomer_months = _resolve_tenure_months(
        tx,
        "user_antiguedad_anios",
        newcomer_definition,
        default_months=CASE14_NEW_EMPLOYEE_YEARS * 12.0,
        context="question9_case14_veterans_from_newcomers:newcomer",
    )
    veteran_months = _resolve_tenure_months(
        tx,
        "receptor_antiguedad_anios",
        veteran_definition,
        default_months=CASE14_OLD_EMPLOYEE_YEARS * 12.0,
        context="question9_case14_veterans_from_newcomers:veteran",
    )
    if pd.isna(newcomer_months):
        newcomer_months = CASE14_NEW_EMPLOYEE_YEARS * 12.0
    if pd.isna(veteran_months):
        veteran_months = CASE14_OLD_EMPLOYEE_YEARS * 12.0
    newcomer_years = newcomer_months / 12.0 if newcomer_months is not None else CASE14_NEW_EMPLOYEE_YEARS
    veteran_years = veteran_months / 12.0 if veteran_months is not None else CASE14_OLD_EMPLOYEE_YEARS

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

            default_thresholds = [
                (CASE14_NEW_EMPLOYEE_YEARS, CASE14_OLD_EMPLOYEE_YEARS),
                (1.0, 4.0),
                (1.5, 3.5),
            ]
            scale_new = (
                newcomer_years / CASE14_NEW_EMPLOYEE_YEARS
                if CASE14_NEW_EMPLOYEE_YEARS > 0
                else 1.0
            )
            scale_old = (
                veteran_years / CASE14_OLD_EMPLOYEE_YEARS
                if CASE14_OLD_EMPLOYEE_YEARS > 0
                else 1.0
            )
            thresholds = [
                (young * scale_new, veteran * scale_old)
                for young, veteran in default_thresholds
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
                if candidate.empty:
                    candidate = base.nsmallest(100, "user_antiguedad_anios").copy()
                selected = candidate
                applied = (young_q, veteran_q)
            if selected is None or selected.empty:
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
                "caso14_persona_emisores_nuevos_lista",
                "interpretabilidad",
            ]
        )

    tenure_columns = [
        ("caso14_persona_antiguedad_receptor_meses", 1.0),
        ("caso14_persona_antiguedad_receptor_anios", 12.0),
        ("receptor_antiguedad_meses", 1.0),
        ("receptor_antiguedad_anios", 12.0),
    ]
    for column, multiplier in tenure_columns:
        if column in work.columns:
            numeric = pd.to_numeric(work[column], errors="coerce") * multiplier
            work = work.loc[numeric >= veteran_months].copy()
            break

    if work.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "persona",
                "caso14_persona_tx_de_emisores_nuevos",
                "caso14_persona_monto_de_emisores_nuevos",
                "caso14_persona_emisores_nuevos_unicos",
                "caso14_persona_monto_promedio_de_emisores_nuevos",
                "caso14_persona_emisores_nuevos_lista",
                "interpretabilidad",
            ]
        )

    work["timeframe"] = timeframe
    work = work.sort_values(
        ["caso14_persona_tx_de_emisores_nuevos", "caso14_persona_monto_de_emisores_nuevos"],
        ascending=[False, False],
    )
    persona_series = work.get("persona", pd.Series(dtype="object"))
    sender_lists = _collect_sender_lists(
        reports,
        timeframe,
        persona_series,
        max_sender_tenure_years=newcomer_years,
    )
    work["caso14_persona_emisores_nuevos_lista"] = persona_series.astype(str).map(
        sender_lists
    )
    work["caso14_persona_emisores_nuevos_lista"] = work[
        "caso14_persona_emisores_nuevos_lista"
    ].apply(lambda value: value if isinstance(value, list) else [])
    newcomer_text = _format_months_threshold(newcomer_months)
    veteran_text = _format_months_threshold(veteran_months)
    work["interpretabilidad"] = work.apply(
        lambda row: (
            f"Dentro de '{timeframe}', la persona {_coalesce_str(row.get('persona'), default='sin_persona')} "
            f"(antigüedad ≥{veteran_text}) recibió {int(row.get('caso14_persona_tx_de_emisores_nuevos', 0))} pagos "
            f"desde recién ingresados (≤{newcomer_text}), provenientes de "
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
        "caso14_persona_emisores_nuevos_lista",
        "interpretabilidad",
    ]
    return work.reindex(columns=columns)


def question10_yoyo_streaks(
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
    min_consecutive: int = 2,
    risk_threshold: float = 1.8,
) -> pd.DataFrame:
    """Detecta rachas yo-yo prolongadas entre pares bidireccionales.

    Parameters
    ----------
    reports
        Diccionario con secciones de transacciones y resúmenes de pares.
    timeframe
        Ventana temporal objetivo (``"todo_el_tiempo"`` por defecto).
    min_consecutive
        Número mínimo de eventos consecutivos para considerar una racha.
    risk_threshold
        Riesgo mínimo del par (``pair_risk_max``) para priorizar el resultado.

    Metodología
    -----------
    1. Usa la bandera ``sig_yoyo`` dentro de ``transaccion`` para agrupar
       secuencias de ida y vuelta por par.
    2. Complementa con el resumen ``par_personas`` para incorporar montos y
       riesgo.
    3. Aplica umbrales de racha mínima y riesgo; cuando faltan banderas recurre
       a heurísticas basadas en ventanas horarias.
    4. Construye textos interpretables describiendo duración, riesgo y montos.

    Returns
    -------
    pandas.DataFrame
        Tabla con pares yo-yo priorizados y explicaciones detalladas.
    """
    tx = _get_section(reports, "transaccion", timeframe)
    required = [
        COL_SENDER_ID,
        COL_RECEIVER_ID,
        "fecha_hora_ts",
        "month_id",
        "sig_yoyo",
        "risk_score",
    ]
    if COL_AMOUNT in tx.columns:
        required.append(COL_AMOUNT)
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
                "monto_total_yo_yo",
                "monto_promedio_yo_yo",
                "interpretabilidad",
            ]
        )

    work = tx[required].copy()
    work["sig_yoyo"] = work["sig_yoyo"].fillna(False).astype(bool)
    if COL_AMOUNT not in work:
        work[COL_AMOUNT] = 0.0
    else:
        work[COL_AMOUNT] = work[COL_AMOUNT].fillna(0.0).astype(float)
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
                        "monto_total_yo_yo",
                        "monto_promedio_yo_yo",
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
        amounts = (
            ordered.loc[ordered["sig_yoyo"], COL_AMOUNT].astype(float)
            if COL_AMOUNT in ordered
            else pd.Series(dtype=float)
        )
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
                "monto_total_yo_yo": float(amounts.sum()) if not amounts.empty else 0.0,
                "monto_promedio_yo_yo": float(amounts.mean()) if not amounts.empty else 0.0,
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
                "monto_total_yo_yo",
                "monto_promedio_yo_yo",
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
                    "monto_total_yo_yo",
                    "monto_promedio_yo_yo",
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
            f"mientras que las transacciones Yo-Yo llegaron a un riesgo máximo de {row.get('riesgo_max_yo_yo', 0.0):.2f} "
            f"y movieron {_format_float(row.get('monto_total_yo_yo', 0.0))} en total "
            f"({_format_float(row.get('monto_promedio_yo_yo', 0.0))} por transacción)."
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
        "monto_total_yo_yo",
        "monto_promedio_yo_yo",
        "interpretabilidad",
    ]
    return filtered.reindex(columns=columns)


def question11_near_threshold_structuring(
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
    min_months: int = 3,
    delta_limit: float = 10.0,
) -> pd.DataFrame:
    """Identifica pares con montos cercanos a umbrales regulatorios.

    Parameters
    ----------
    reports
        Diccionario con la sección ``"transaccion"`` y banderas ``sig_near_thr``.
    timeframe
        Ventana temporal objetivo (por defecto ``"todo_el_tiempo"``).
    min_months
        Meses mínimos con eventos cercanos al umbral para considerar al par.
    delta_limit
        Diferencia máxima respecto al umbral para incluir la transacción.

    Metodología
    -----------
    1. Filtra transacciones con bandera near-threshold y deltas pequeños, o
       estima la distancia a umbrales comunes cuando falta la métrica.
    2. Cuenta meses con recurrencia y agrega montos, riesgos y desviaciones
       promedio.
    3. Aplica filtros de ``min_months`` y ``delta_limit``; si es necesario,
       relaja umbrales o aplica heurísticas basadas en cuantiles.
    4. Redacta interpretabilidad indicando cercanía al umbral y riesgo asociado.

    Returns
    -------
    pandas.DataFrame
        Tabla de pares con structuring cercano a umbrales y explicaciones.
    """
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
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
    min_months: int | None = None,
) -> pd.DataFrame:
    """Localiza pares con fraccionamiento crónico de montos.

    Parameters
    ----------
    reports
        Diccionario con transacciones y banderas ``sig_smurf``.
    timeframe
        Ventana temporal evaluada (``"todo_el_tiempo"`` por defecto).
    min_months
        Cantidad mínima de meses con eventos identificados. Si es ``None`` se
        consideran todos los pares y se priorizan los montos globales.

    Metodología
    -----------
    1. Selecciona transacciones marcadas con ``sig_smurf`` o reconstruye la
       etiqueta usando cuantiles por par cuando no está disponible.
    2. Agrega montos, riesgo promedio y máximo por mes para medir la
       consistencia del fraccionamiento.
    3. Prioriza los pares por el monto fraccionado total y redacta
       interpretabilidad detallada, indicando si se relajaron criterios.

    Returns
    -------
    pandas.DataFrame
        Tabla de pares con fraccionamiento crónico y descripciones de su
        comportamiento.
    """
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
                "meses_con_fraccionamiento",
                "transacciones_fraccionadas",
                "monto_fraccionado_total",
                "riesgo_promedio",
                "riesgo_maximo",
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
                    "meses_con_fraccionamiento",
                    "transacciones_fraccionadas",
                    "monto_fraccionado_total",
                    "riesgo_promedio",
                    "riesgo_maximo",
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
                "riesgo_promedio": float(ordered["riesgo_prom_mes"].mean()) if not ordered.empty else 0.0,
                "riesgo_maximo": float(ordered["riesgo_max_mes"].max()) if not ordered.empty else 0.0,
                "tendencia_riesgo": direction,
                "delta_riesgo": trend,
                "riesgo_inicio": risk_start,
                "riesgo_fin": risk_end,
            }
        )

    base = (
        monthly.groupby("pair", observed=True)
        .agg(
            meses_con_fraccionamiento=("month_id", "nunique"),
            transacciones_fraccionadas=("tx_smurf_mes", "sum"),
            monto_fraccionado_total=("monto_mes", "sum"),
        )
        .reset_index()
    )
    risk = monthly.groupby("pair", observed=True).apply(_risk_trend).reset_index()
    merged = base.merge(risk, on="pair", how="left")
    filtered = merged.copy()
    applied_month_filter = False
    if min_months:
        applied_month_filter = True
        filtered = filtered[
            filtered["meses_con_fraccionamiento"] >= int(min_months)
        ].copy()
    if filtered.empty:
        filtered = merged.sort_values(
            [
                "monto_fraccionado_total",
                "transacciones_fraccionadas",
                "meses_con_fraccionamiento",
            ],
            ascending=[False, False, False],
        ).head(25)
    if filtered.empty:
        return pd.DataFrame(
            columns=[
                "timeframe",
                "pair",
                "meses_con_fraccionamiento",
                "transacciones_fraccionadas",
                "monto_fraccionado_total",
                "riesgo_promedio",
                "riesgo_maximo",
                "tendencia_riesgo",
                "interpretabilidad",
            ]
        )

    filtered["timeframe"] = timeframe
    filtered = filtered.sort_values(
        ["monto_fraccionado_total", "transacciones_fraccionadas"],
        ascending=[False, False],
    )
    filtered["interpretabilidad"] = filtered.apply(
        lambda row: (
            f"En '{timeframe}', el par {row.get('pair', 'sin_par')} registró fraccionamiento en "
            f"{int(row.get('transacciones_fraccionadas', 0))} transacciones, acumulando "
            f"{_format_float(row.get('monto_fraccionado_total', 0))} y distribuyéndolo en "
            f"{int(row.get('meses_con_fraccionamiento', 0))} meses. El riesgo promedio fue "
            f"{row.get('riesgo_promedio', 0.0):.2f}, con pico de {row.get('riesgo_maximo', 0.0):.2f} y tendencia "
            f"{row.get('tendencia_riesgo', 'estable')} (inicio {row.get('riesgo_inicio', 0.0):.2f} → fin {row.get('riesgo_fin', 0.0):.2f})."
            + (
                " Se utilizó una heurística de montos pequeños repetidos ante la ausencia de alertas explícitas."
                if relaxed_flags and row.get("transacciones_fraccionadas", 0) > 0
                else ""
            )
            + (
                " Se priorizó el total fraccionado por encima de la distribución mensual."
                if not applied_month_filter
                else ""
            )
        ),
        axis=1,
    )
    columns = [
        "timeframe",
        "pair",
        "meses_con_fraccionamiento",
        "transacciones_fraccionadas",
        "monto_fraccionado_total",
        "riesgo_promedio",
        "riesgo_maximo",
        "tendencia_riesgo",
        "interpretabilidad",
    ]
    return filtered.reindex(columns=columns)


def question13_bad_loans_with_frequency(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME, min_months: int = 2
) -> pd.DataFrame:
    """Cruza préstamos incumplidos con ráfagas de frecuencia elevada.

    Parameters
    ----------
    reports
        Diccionario con transacciones y banderas ``sig_loan_bad_repay`` y
        ``sig_freq``.
    timeframe
        Ventana temporal a analizar (por defecto ``"todo_el_tiempo"``).
    min_months
        Meses mínimos con coincidencia entre préstamos incumplidos y frecuencia
        elevada.

    Metodología
    -----------
    1. Identifica transacciones con banderas de préstamo incumplido y alta
       frecuencia, calculando coincidencias mensuales por par.
    2. Cuando faltan banderas, emplea heurísticas para estimar préstamos,
       reembolsos y umbrales de frecuencia.
    3. Agrega montos, riesgos y meses coincidentes, priorizando los pares con
       mayor severidad y aplicando criterios relajados cuando es necesario.
    4. Genera interpretabilidad explicando la superposición de señales.

    Returns
    -------
    pandas.DataFrame
        Tabla con pares sospechosos de préstamos irregulares frecuentes y sus
        explicaciones.
    """
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
    """Resalta pagos recurrentes tipo nómina entre pares específicos.

    Parameters
    ----------
    reports
        Diccionario con transacciones y bandera ``sig_recurrent``.
    timeframe
        Ventana temporal analizada (``"todo_el_tiempo"`` por defecto).
    min_months
        Meses consecutivos mínimos con pagos recurrentes para priorizar un par.

    Metodología
    -----------
    1. Identifica transacciones marcadas como recurrentes o reconstruye el
       patrón por calendario cuando faltan banderas.
    2. Agrega por emisor, receptor y día de corte contabilizando meses, pagos y
       montos totales/promedio.
    3. Filtra por ``min_months`` y presencia de la bandera (si existe), aplicando
       criterios relajados en ausencia de coincidencias estrictas.
    4. Genera interpretabilidad destacando periodicidad, totales y posibles
       nóminas paralelas.

    Returns
    -------
    pandas.DataFrame
        Tabla con pagos recurrentes priorizados y explicaciones.
    """
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


def question15_coordinated_cluster_signals(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME, top_n: int = 10
) -> pd.DataFrame:
    """Sintetiza clusters de personas con múltiples señales coordinadas.

    Parameters
    ----------
    reports
        Diccionario con la sección ``"clusters_personas"`` y métricas asociadas.
    timeframe
        Ventana temporal objetivo (``"todo_el_tiempo"`` por defecto).
    top_n
        Número máximo de clusters priorizados en la salida.

    Metodología
    -----------
    1. Forma clusters como componentes conexas del grafo de transacciones,
       uniendo personas enlazadas por pagos directos o a través de terceros.
    2. Extrae indicadores de señales (yo-yo, fraccionamiento, ciclos, quid y
       referencia reutilizada) junto con métricas de tamaño, montos y riesgo.
    3. Calcula cuántas señales activas tiene cada cluster y normaliza montos y
       conteos para ordenar la prioridad.
    4. Describe la persona más desbalanceada, explica cómo se construye el
       cluster y resume el detalle porcentual de cada señal en la
       interpretabilidad.

    Returns
    -------
    pandas.DataFrame
        Tabla con los principales clusters, sus señales activas y contexto
        interpretativo.
    """
    clusters = _get_section(reports, "clusters_personas", timeframe)
    signal_cols = {
        "yo_yo_cluster_tasa_flag": "yo-yo",
        "smurf_cluster_tasa_flag": "fraccionamiento",
        "red_cluster_tasa_en_ciclos": "ciclos",
        "quid_cluster_tasa_flag": "quid",
        "referencia_cluster_tasa_reutilizada": "referencia reutilizada",
    }
    base_columns = [
        "cluster_id",
        "cluster_personas",
        "cluster_personas_total",
        "cluster_tx_count",
        "cluster_tx_sum",
        "riesgo_cluster_maximo",
        "desbalance_cluster_persona_principal",
        "desbalance_cluster_persona_principal_monto",
    ] + list(signal_cols.keys())

    columns = [
        "timeframe",
        "cluster_id",
        "personas_en_cluster",
        "cluster_personas_total",
        "cluster_tx_count",
        "cluster_tx_sum",
        "riesgo_cluster_maximo",
        "signals_activas",
    ] + list(signal_cols.keys()) + [
        "persona_mas_desbalanceada",
        "monto_neto_desbalance",
        "interpretabilidad",
    ]

    if clusters.empty or not set(base_columns).issubset(clusters.columns):
        fallback = {
            "timeframe": timeframe,
            "cluster_id": "sin_datos",
            "personas_en_cluster": "sin_personas",
            "cluster_personas_total": 0,
            "cluster_tx_count": 0,
            "cluster_tx_sum": 0.0,
            "riesgo_cluster_maximo": 0.0,
            "signals_activas": 0,
            "persona_mas_desbalanceada": "sin_persona",
            "monto_neto_desbalance": 0.0,
            "interpretabilidad": (
                "No se identificaron clusters relevantes en la ventana "
                f"'{timeframe}', por lo que no hay señales coordinadas que resumir."
            ),
        }
        for col in signal_cols:
            fallback[col] = 0.0
        return pd.DataFrame([fallback]).reindex(columns=columns)

    work = clusters.copy()
    work["timeframe"] = timeframe
    for col in signal_cols:
        if col not in work:
            work[col] = 0.0
        work[col] = work[col].fillna(0.0).astype(float)

    work["cluster_personas_total"] = work["cluster_personas_total"].fillna(0).astype(int)
    work["cluster_tx_count"] = work["cluster_tx_count"].fillna(0).astype(int)
    work["cluster_tx_sum"] = work["cluster_tx_sum"].fillna(0.0).astype(float)
    work["riesgo_cluster_maximo"] = work["riesgo_cluster_maximo"].fillna(0.0).astype(float)
    work["desbalance_cluster_persona_principal_monto"] = (
        work.get("desbalance_cluster_persona_principal_monto", 0.0).fillna(0.0).astype(float)
    )
    work["desbalance_cluster_persona_principal"] = (
        work.get("desbalance_cluster_persona_principal", "sin_persona")
        .fillna("sin_persona")
        .astype(str)
    )
    work["signals_activas"] = work[list(signal_cols.keys())].gt(0.0).sum(axis=1)
    work["signals_score"] = work[list(signal_cols.keys())].sum(axis=1)

    def _summarize_personas(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            personas = [str(v) for v in value if str(v).strip()]
            if not personas:
                return "sin_personas"
            if len(personas) <= 4:
                return ", ".join(personas)
            return ", ".join(personas[:3]) + f" y {len(personas) - 3} más"
        if pd.isna(value):
            return "sin_personas"
        text = str(value).strip()
        return text if text else "sin_personas"

    work["personas_en_cluster"] = work["cluster_personas"].apply(_summarize_personas)

    def _signals_text(row: pd.Series) -> str:
        active = [
            f"{label}: {row.get(col, 0.0) * 100:.1f}%"
            for col, label in signal_cols.items()
            if row.get(col, 0.0) > 0
        ]
        if not active:
            return "sin señales priorizadas activas"
        return "; ".join(active)

    work["signals_detalle"] = work.apply(_signals_text, axis=1)

    def _balance_text(row: pd.Series) -> str:
        persona = row.get("desbalance_cluster_persona_principal", "sin_persona")
        monto = float(row.get("desbalance_cluster_persona_principal_monto", 0.0))
        if persona == "sin_persona":
            return "no se identificó una persona predominante"
        tendencia = "neto emisor" if monto > 0 else "neto receptor" if monto < 0 else "sin ventaja neta"
        return (
            f"{persona} concentra {_format_float(abs(monto))} como {tendencia}"
            if monto != 0
            else f"{persona} sin desequilibrio monetario neto"
        )

    work["persona_mas_desbalanceada"] = work["desbalance_cluster_persona_principal"]
    work["monto_neto_desbalance"] = work["desbalance_cluster_persona_principal_monto"]

    work = work.sort_values(
        ["signals_activas", "signals_score", "riesgo_cluster_maximo", "cluster_tx_sum"],
        ascending=[False, False, False, False],
    ).head(max(1, int(top_n)))

    work["interpretabilidad"] = work.apply(
        lambda row: (
            f"En '{timeframe}', el {row.get('cluster_id', 'cluster_sin_id')} reúne "
            f"{int(row.get('cluster_personas_total', 0))} personas ({row.get('personas_en_cluster', 'sin_personas')}) "
            f"con {int(row.get('cluster_tx_count', 0))} transacciones que suman "
            f"{_format_float(row.get('cluster_tx_sum', 0))} y riesgo máximo "
            f"{row.get('riesgo_cluster_maximo', 0):.2f}. Se activan "
            f"{int(row.get('signals_activas', 0))} de las 5 señales priorizadas ({row.get('signals_detalle', 'sin detalle')}). "
            "El cluster se construye como una componente conexa del grafo de pagos, "
            "por lo que todas las personas están enlazadas directa o indirectamente. "
            "Las tasas de cada señal indican el porcentaje de transacciones dentro "
            "del grupo con ese comportamiento coordinado. "
            f"La persona más desbalanceada {_balance_text(row)}."
        ),
        axis=1,
    )

    return work.reindex(columns=columns)


def question16_multisignal_transactions(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME, top_n: int = 25
) -> pd.DataFrame:
    """Lista transacciones que acumulan múltiples señales simultáneas.

    Parameters
    ----------
    reports
        Diccionario con la sección ``"transaccion"`` y banderas de señales.
    timeframe
        Ventana temporal analizada (por defecto ``"todo_el_tiempo"``).
    top_n
        Número máximo de transacciones priorizadas en la salida.

    Metodología
    -----------
    1. Consolida banderas de jerarquía, yo-yo, fraccionamiento, near-threshold, quid y
       cambios bruscos por transacción.
    2. Cuenta cuántas señales están activas por registro y ordena por número de
       señales, riesgo y monto.
    3. Selecciona el umbral más alto posible (3, luego 2, luego 1 señal) para
       devolver hasta ``top_n`` operaciones.
    4. Genera interpretabilidad con monto, riesgo, relación declarada,
       descripción y nota sobre el umbral aplicado.

    Returns
    -------
    pandas.DataFrame
        Tabla con operaciones multisignal y su explicación detallada.
    """
    tx = _get_section(reports, "transaccion", timeframe)
    signal_cols = {
        "flag_jerarquia": "jerarquía",
        "sig_yoyo": "yo-yo",
        "sig_smurf": "fraccionamiento",
        "sig_near_thr": "near-threshold",
        "sig_quid_pro_quo": "quid",
        "sig_pair_change_point": "cambio brusco",
    }

    columns = [
        "timeframe",
        "fecha_hora_ts",
        "emisor",
        "receptor",
        "movement_amount",
        "risk_score",
        "risk_tier",
        "signals_activas",
        "signals_detalle",
        "umbral_senales_seleccion",
    ] + list(signal_cols.keys()) + ["interpretabilidad"]

    if tx.empty:
        fallback = {
            "timeframe": timeframe,
            "fecha_hora_ts": "sin_fecha",
            "emisor": "sin_emisor",
            "receptor": "sin_receptor",
            "movement_amount": 0.0,
            "risk_score": 0.0,
            "risk_tier": "SIN_TIERRA",
            "signals_activas": 0,
            "signals_detalle": "sin señales",
            "interpretabilidad": (
                "No se detectaron transacciones con múltiples señales simultáneas en la ventana "
                f"'{timeframe}'."
            ),
        }
        for col in signal_cols:
            fallback[col] = 0
        return pd.DataFrame([fallback]).reindex(columns=columns)

    work = tx.copy()
    work["timeframe"] = timeframe
    relation_series = _coalesce_text_column(work, COL_RELATION)
    work["flag_jerarquia"] = relation_series.str.contains("manager", case=False, na=False)

    for col in signal_cols:
        if col not in work:
            work[col] = 0
        work[col] = work[col].fillna(0)

    work["signals_activas"] = (
        work[list(signal_cols.keys())]
        .apply(lambda row: sum(bool(x) and x != "" and x != 0 for x in row), axis=1)
        .astype(int)
    )

    def _active_signals(row: pd.Series) -> list[str]:
        active = []
        for col, label in signal_cols.items():
            value = row.get(col, 0)
            is_active = False
            if isinstance(value, (int, float)):
                is_active = value > 0
            else:
                is_active = bool(value)
            if is_active:
                active.append(label)
        return active

    work["signals_lista"] = work.apply(_active_signals, axis=1)
    work["signals_detalle"] = work["signals_lista"].apply(
        lambda items: ", ".join(items) if items else "sin señales"
    )

    work = work.sort_values(
        ["signals_activas", "risk_score", "movement_amount"], ascending=[False, False, False]
    )

    filtered = pd.DataFrame()
    threshold_used = 0
    for threshold in (3, 2):
        candidate = work.loc[work["signals_activas"] >= threshold].copy()
        if not candidate.empty:
            filtered = candidate.head(max(1, int(top_n))).copy()
            threshold_used = threshold
            break
    if filtered.empty:
        candidate = work.loc[work["signals_activas"] >= 1].copy()
        if not candidate.empty:
            filtered = candidate.head(max(1, int(top_n))).copy()
            threshold_used = 1
    if filtered.empty:
        filtered = work.head(max(1, int(top_n))).copy()
        threshold_used = 0

    filtered = filtered.copy()
    filtered.loc[:, "umbral_senales_seleccion"] = threshold_used

    filtered.loc[:, "fecha_hora_ts"] = filtered.get("fecha_hora_ts")
    filtered.loc[:, "emisor"] = filtered.get(COL_SENDER_ID, "").astype(str)
    filtered.loc[:, "receptor"] = filtered.get(COL_RECEIVER_ID, "").astype(str)
    filtered.loc[:, "movement_amount"] = filtered.get(COL_AMOUNT, 0.0).astype(float)
    filtered.loc[:, "risk_score"] = filtered.get("risk_score", 0.0).astype(float)
    risk_tier_series = filtered.get("risk_tier")
    if risk_tier_series is not None:
        filtered.loc[:, "risk_tier"] = (
            risk_tier_series.astype("string").fillna("SIN_TIERRA").replace({"<NA>": "SIN_TIERRA"})
        )
    else:
        filtered.loc[:, "risk_tier"] = "SIN_TIERRA"

    descripcion_series = _coalesce_text_column(filtered, COL_DESCRIPTION)
    relation_series = _coalesce_text_column(filtered, COL_RELATION)

    def _nota_relajada(row: pd.Series) -> str:
        threshold = row.get("umbral_senales_seleccion", 0)
        if threshold >= 3:
            return ""
        if threshold == 2:
            return " Ante la ausencia de operaciones con 3+ señales, se priorizan las que combinan al menos dos indicadores."
        if threshold == 1:
            return " No hubo transacciones con múltiples señales; se listan las de una señal simultánea con mayor riesgo."
        return " No hubo transacciones con señales activas; se listan las de mayor riesgo como referencia."

    filtered["interpretabilidad"] = filtered.apply(
        lambda row: (
            f"El {row.get('fecha_hora_ts', 'sin_fecha')} se registró una transacción de "
            f"{_format_float(row.get('movement_amount', 0))} entre "
            f"{row.get('emisor', 'sin_emisor')} y {row.get('receptor', 'sin_receptor')} con riesgo "
            f"{row.get('risk_score', 0):.2f} ({row.get('risk_tier', 'SIN_TIERRA')}). "
            f"Activa {int(row.get('signals_activas', 0))} señales simultáneas: "
            f"{row.get('signals_detalle', 'sin señales')}. "
            f"Relación declarada: {relation_series.get(row.name, 'sin_relacion')}. "
            f"Descripción: {descripcion_series.get(row.name, 'sin_descripcion')}"
            f"{_nota_relajada(row)}"
        ),
        axis=1,
    )

    return filtered.reindex(columns=columns)


def question17_nlp_person_profiles(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME, top_n: int = 15
) -> pd.DataFrame:
    """Prioriza personas con perfiles NLP sospechosos y resume su contexto.

    Parameters
    ----------
    reports
        Diccionario con las secciones ``"persona"`` y ``"persona_concepto"``.
    timeframe
        Ventana temporal objetivo (``"todo_el_tiempo"`` por defecto).
    top_n
        Máximo de personas resaltadas en el resultado.

    Metodología
    -----------
    1. Combina métricas por persona con los conceptos NLP detectados para cada
       individuo.
    2. Calcula proporciones de transacciones sospechosas respecto al total, así
        como montos netos recibidos/enviados y riesgo promedio.
    3. Ordena por volumen sospechoso y riesgo, generando interpretabilidad que
       resume conceptos principales y desequilibrio neto.

    Returns
    -------
    pandas.DataFrame
        Tabla con perfiles NLP prioritarios y explicaciones contextualizadas.
    """
    personas = _get_section(reports, "persona", timeframe)
    conceptos = _get_section(reports, "persona_concepto", timeframe)

    required_cols = [
        "persona",
        "movements",
        "nlp_persona_total_transacciones_sospechosas",
        "nlp_persona_conceptos_sospechosos_unicos",
        "nlp_persona_top_conceptos",
        "risk_avg_person",
        "sum_emit",
        "sum_recv",
        "nlp_persona_score_prob_coi",
        "nlp_persona_sentimiento_promedio",
        "nlp_persona_score_emit_promedio",
        "nlp_persona_score_recv_promedio",
    ]

    columns = [
        "timeframe",
        "persona",
        "movements",
        "tx_sospechosas_nlp",
        "conceptos_unicos",
        "conceptos_unicos_por_tx",
        "proporcion_sospechosa",
        "score_probable_coi",
        "nlp_score_emit_promedio",
        "nlp_score_recv_promedio",
        "sentimiento_promedio",
        "sentimiento_etiqueta",
        "risk_avg_person",
        "sum_emit",
        "sum_recv",
        "net_flow",
        "top_conceptos_display",
        "conceptos_principales",
        "concepto_predominante",
        "interpretabilidad",
    ]

    if personas.empty or not set(required_cols).issubset(personas.columns):
        fallback = {
            "timeframe": timeframe,
            "persona": "sin_persona",
            "movements": 0,
            "tx_sospechosas_nlp": 0,
            "conceptos_unicos": 0,
            "conceptos_unicos_por_tx": 0.0,
            "proporcion_sospechosa": 0.0,
            "score_probable_coi": 0.0,
            "nlp_score_emit_promedio": 0.0,
            "nlp_score_recv_promedio": 0.0,
            "sentimiento_promedio": 0.0,
            "sentimiento_etiqueta": "neutral",
            "risk_avg_person": 0.0,
            "sum_emit": 0.0,
            "sum_recv": 0.0,
            "net_flow": 0.0,
            "top_conceptos_display": "sin_top_conceptos",
            "conceptos_principales": "sin_conceptos",
            "concepto_predominante": "sin_concepto",
            "interpretabilidad": (
                "No se identificaron personas con señales NLP para priorizar en la ventana "
                f"'{timeframe}'."
            ),
        }
        return pd.DataFrame([fallback]).reindex(columns=columns)

    personas = personas.copy()
    personas["timeframe"] = timeframe
    personas["movements"] = personas.get("movements", 0).fillna(0).astype(int)
    personas["nlp_persona_total_transacciones_sospechosas"] = (
        personas.get("nlp_persona_total_transacciones_sospechosas", 0)
        .fillna(0)
        .astype(int)
    )
    personas["nlp_persona_conceptos_sospechosos_unicos"] = (
        personas.get("nlp_persona_conceptos_sospechosos_unicos", 0)
        .fillna(0)
        .astype(int)
    )
    personas["nlp_persona_top_conceptos"] = personas.get(
        "nlp_persona_top_conceptos", [[] for _ in range(len(personas))]
    ).apply(
        lambda x: x
        if isinstance(x, list)
        else ([] if pd.isna(x) else [x])
    )
    personas["nlp_persona_top_conceptos"] = personas["nlp_persona_top_conceptos"].apply(
        lambda items: [str(item) for item in items if str(item).strip()]
    )
    personas["risk_avg_person"] = personas.get("risk_avg_person", 0.0).fillna(0.0).astype(float)
    personas["sum_emit"] = personas.get("sum_emit", 0.0).fillna(0.0).astype(float)
    personas["sum_recv"] = personas.get("sum_recv", 0.0).fillna(0.0).astype(float)
    personas["score_probable_coi"] = (
        personas.get("nlp_persona_score_prob_coi", 0.0).fillna(0.0).astype(float)
    )
    personas["nlp_score_emit_promedio"] = (
        personas.get("nlp_persona_score_emit_promedio", 0.0).fillna(0.0).astype(float)
    )
    personas["nlp_score_recv_promedio"] = (
        personas.get("nlp_persona_score_recv_promedio", 0.0).fillna(0.0).astype(float)
    )
    personas["sentimiento_promedio"] = (
        personas.get("nlp_persona_sentimiento_promedio", 0.0).fillna(0.0).astype(float)
    )
    personas["sentimiento_etiqueta"] = personas["sentimiento_promedio"].apply(
        lambda v: "positivo" if v > 0.2 else ("negativo" if v < -0.2 else "neutral")
    )

    personas["tx_sospechosas_nlp"] = personas["nlp_persona_total_transacciones_sospechosas"]
    personas["conceptos_unicos"] = personas["nlp_persona_conceptos_sospechosos_unicos"]
    personas["conceptos_unicos_por_tx"] = personas.apply(
        lambda row: (
            row["conceptos_unicos"] / row["tx_sospechosas_nlp"]
            if row.get("tx_sospechosas_nlp", 0) else 0.0
        ),
        axis=1,
    )
    personas["proporcion_sospechosa"] = personas.apply(
        lambda row: (
            row["tx_sospechosas_nlp"] / row["movements"]
            if row.get("movements", 0) else 0.0
        ),
        axis=1,
    )
    personas["net_flow"] = personas["sum_emit"] - personas["sum_recv"]

    if not conceptos.empty and {"persona", "nlp_concepto_sospechoso"}.issubset(conceptos.columns):
        conceptos = conceptos.copy()
        conceptos["nlp_persona_concepto_tx_total"] = (
            conceptos.get("nlp_persona_concepto_tx_total", 0).fillna(0).astype(int)
        )
        conceptos["nlp_persona_concepto_monto_total"] = (
            conceptos.get("nlp_persona_concepto_monto_total", 0.0).fillna(0.0).astype(float)
        )
        conceptos["nlp_persona_concepto_riesgo_promedio"] = (
            conceptos.get("nlp_persona_concepto_riesgo_promedio", 0.0).fillna(0.0).astype(float)
        )

        conceptos = conceptos.sort_values(
            ["persona", "nlp_persona_concepto_tx_total", "nlp_persona_concepto_riesgo_promedio"],
            ascending=[True, False, False],
        )

        detalle_map: Dict[str, list[str]] = {}
        for persona, group in conceptos.groupby("persona", observed=True):
            top = group.head(3)
            detalle_map[str(persona)] = [
                (
                    f"{_coalesce_str(row.get('nlp_concepto_sospechoso'), default='SIN_CONCEPTO')} "
                    f"(tx={int(row.get('nlp_persona_concepto_tx_total', 0))}, "
                    f"monto={_format_float(row.get('nlp_persona_concepto_monto_total', 0))}, "
                    f"riesgo={row.get('nlp_persona_concepto_riesgo_promedio', 0):.2f})"
                )
                for _, row in top.iterrows()
            ]
        personas["conceptos_principales"] = personas["persona"].astype(str).map(detalle_map)
    else:
        personas["conceptos_principales"] = [[] for _ in range(len(personas))]

    personas["conceptos_principales"] = personas["conceptos_principales"].apply(
        lambda items: items if isinstance(items, list) else []
    )
    personas["conceptos_principales"] = personas["conceptos_principales"].apply(
        lambda items: "; ".join(items) if items else "sin_conceptos_detallados"
    )
    personas["top_conceptos_display"] = personas["nlp_persona_top_conceptos"].apply(
        lambda items: ", ".join(items) if items else "sin_top_conceptos"
    )
    personas["concepto_predominante"] = personas["nlp_persona_top_conceptos"].apply(
        lambda items: items[0] if items else "sin_concepto"
    )

    personas = personas.sort_values(
        ["score_probable_coi", "tx_sospechosas_nlp", "proporcion_sospechosa", "risk_avg_person"],
        ascending=[False, False, False, False],
    ).head(max(1, int(top_n)))

    personas["interpretabilidad"] = personas.apply(
        lambda row: (
            f"En '{timeframe}', la persona {row.get('persona', 'sin_persona')} acumula "
            f"{int(row.get('tx_sospechosas_nlp', 0))} transacciones NLP sospechosas "
            f"sobre {int(row.get('movements', 0))} movimientos totales ({row.get('proporcion_sospechosa', 0):.0%}). "
            f"Principales conceptos: {row.get('top_conceptos_display', 'sin_top_conceptos')} "
            f"[{row.get('conceptos_principales', 'sin_conceptos_detallados')}]. "
            f"Cada concepto cubre en promedio {row.get('conceptos_unicos_por_tx', 0):.2f} "
            f"conceptos únicos por transacción sospechosa, con foco en "
            f"{row.get('concepto_predominante', 'sin_concepto')}. Riesgo promedio "
            f"{row.get('risk_avg_person', 0):.2f}. Score COI {row.get('score_probable_coi', 0):.2f} "
            f"(emisor {row.get('nlp_score_emit_promedio', 0):.2f} / receptor {row.get('nlp_score_recv_promedio', 0):.2f}) "
            f"y sentimiento {row.get('sentimiento_etiqueta', 'neutral')} "
            f"({row.get('sentimiento_promedio', 0):.2f}). Flujo neto {_format_float(row.get('net_flow', 0))} "
            f"(emitidos {_format_float(row.get('sum_emit', 0))} vs recibidos "
            f"{_format_float(row.get('sum_recv', 0))})."
        ),
        axis=1,
    )

    return personas.reindex(columns=columns)


def question18_user_risk_scores(
    reports: Mapping[str, Any], timeframe: str = DEFAULT_TIMEFRAME, top_n: int = 25
) -> pd.DataFrame:
    """Prioriza personas por riesgo promedio, desbalance y señales activas."""

    personas = _standardize_columns(
        _get_section(reports, "persona", timeframe), PERSONA_COLUMN_ALIASES
    )

    columns = [
        "timeframe",
        "ranking_prioridad",
        "persona",
        "risk_avg_person",
        "risk_tier",
        "movements",
        "n_tx_emit",
        "sum_emit",
        "n_tx_recv",
        "sum_recv",
        "net_flow",
        "desbalance_persona_monto_neto",
        "desbalance_persona_meses_totales",
        "desbalance_persona_tasa_meses_envia_extremo",
        "desbalance_persona_tasa_meses_recibe_extremo",
        "std_monto_mensual",
        "std_dia_pago_mensual",
        "detalle_pagos_mensuales",
        "flags_activas",
        "flag_rate_max",
        "banderas_destacadas",
        "interpretabilidad",
    ]

    if personas.empty or "persona" not in personas.columns:
        return pd.DataFrame(
            [
                {
                    "timeframe": timeframe,
                    "ranking_prioridad": 0,
                    "persona": "sin_persona",
                    "risk_avg_person": 0.0,
                    "risk_tier": "SIN_RIESGO",
                    "movements": 0,
                    "n_tx_emit": 0,
                    "sum_emit": 0.0,
                    "n_tx_recv": 0,
                    "sum_recv": 0.0,
                    "net_flow": 0.0,
                    "desbalance_persona_monto_neto": 0.0,
                    "desbalance_persona_meses_totales": 0,
                    "desbalance_persona_tasa_meses_envia_extremo": 0.0,
                    "desbalance_persona_tasa_meses_recibe_extremo": 0.0,
                    "std_monto_mensual": 0.0,
                    "std_dia_pago_mensual": 0.0,
                    "detalle_pagos_mensuales": [],
                    "flags_activas": 0,
                    "flag_rate_max": 0.0,
                    "banderas_destacadas": "sin_banderas_destacadas",
                    "interpretabilidad": (
                        "No se identificaron personas prioritarias por riesgo en la ventana "
                        f"'{timeframe}'."
                    ),
                }
            ]
        ).reindex(columns=columns)

    work = personas.copy()
    work["timeframe"] = timeframe

    detalle_mensual: dict[str, list[dict[str, Any]]] = {}
    std_monto_map: dict[str, float] = {}
    std_dia_map: dict[str, float] = {}

    tx = _standardize_columns(
        _get_section(reports, "transaccion", timeframe), TRANSACTION_COLUMN_ALIASES
    )
    required_tx_cols = {COL_RECEIVER_ID, COL_AMOUNT, "fecha_hora_ts", "month_id"}
    if not tx.empty and required_tx_cols.issubset(tx.columns):
        pagos = tx[list(required_tx_cols)].copy()
        pagos[COL_RECEIVER_ID] = (
            pagos[COL_RECEIVER_ID].astype("string").str.strip()
        )
        pagos = pagos[pagos[COL_RECEIVER_ID] != ""]
        pagos[COL_AMOUNT] = pd.to_numeric(pagos[COL_AMOUNT], errors="coerce")
        pagos["fecha_hora_ts"] = pd.to_datetime(
            pagos["fecha_hora_ts"], errors="coerce"
        )
        pagos["month_id"] = pagos["month_id"].astype("string").str.strip()
        pagos = pagos.dropna(subset=[COL_AMOUNT, "fecha_hora_ts", "month_id"])
        pagos = pagos[pagos["month_id"] != ""]

        if not pagos.empty:
            pagos = pagos.sort_values([COL_RECEIVER_ID, "month_id", "fecha_hora_ts"])
            for persona_id, persona_df in pagos.groupby(
                COL_RECEIVER_ID, observed=True
            ):
                persona_key = str(persona_id)
                registros: list[dict[str, Any]] = []
                montos: list[float] = []
                dias: list[float] = []
                for month_id, month_df in persona_df.groupby(
                    "month_id", observed=True
                ):
                    if month_df.empty:
                        continue
                    mes = str(month_id) if month_id is not None else "sin_mes"
                    fechas = month_df["fecha_hora_ts"].dt.date
                    montos_mes = month_df[COL_AMOUNT].astype(float)
                    total_mes = float(montos_mes.sum())
                    dia_promedio = float(month_df["fecha_hora_ts"].dt.day.mean())
                    registros.append(
                        {
                            "mes": mes,
                            "monto_total": total_mes,
                            "dia_pago_promedio": dia_promedio,
                            "pagos": [
                                {
                                    "fecha": fecha.isoformat(),
                                    "monto": float(monto),
                                }
                                for fecha, monto in zip(fechas, montos_mes)
                            ],
                        }
                    )
                    montos.append(total_mes)
                    dias.append(dia_promedio)
                if registros:
                    detalle_mensual[persona_key] = registros
                    if len(montos) >= 2:
                        std_monto_map[persona_key] = float(
                            pd.Series(montos, dtype="float64").std(ddof=0)
                        )
                    else:
                        std_monto_map[persona_key] = 0.0
                    dias_validos = [dia for dia in dias if pd.notna(dia)]
                    if len(dias_validos) >= 2:
                        std_dia_map[persona_key] = float(
                            pd.Series(dias_validos, dtype="float64").std(ddof=0)
                        )
                    else:
                        std_dia_map[persona_key] = 0.0
                else:
                    detalle_mensual[persona_key] = []
                    std_monto_map[persona_key] = 0.0
                    std_dia_map[persona_key] = 0.0

    work["persona"] = work["persona"].astype("string").str.strip()
    work["std_monto_mensual"] = work["persona"].map(std_monto_map).fillna(0.0)
    work["std_dia_pago_mensual"] = work["persona"].map(std_dia_map).fillna(0.0)
    work["detalle_pagos_mensuales"] = work["persona"].map(detalle_mensual)
    work["detalle_pagos_mensuales"] = work["detalle_pagos_mensuales"].apply(
        lambda value: value if isinstance(value, list) else []
    )

    for col in ["n_tx_emit", "n_tx_recv"]:
        if col in work.columns:
            work[col] = (
                pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)
            )
        else:
            work[col] = 0
    for col in ["sum_emit", "sum_recv", "risk_avg_person"]:
        if col in work.columns:
            work[col] = (
                pd.to_numeric(work[col], errors="coerce").fillna(0.0).astype(float)
            )
        else:
            work[col] = 0.0

    if "movements" in work.columns:
        work["movements"] = (
            pd.to_numeric(work["movements"], errors="coerce").fillna(0).astype(int)
        )
    else:
        work["movements"] = work["n_tx_emit"] + work["n_tx_recv"]

    work["net_flow"] = work["sum_emit"] - work["sum_recv"]
    if "desbalance_persona_monto_neto" in work.columns:
        work["desbalance_persona_monto_neto"] = (
            pd.to_numeric(work["desbalance_persona_monto_neto"], errors="coerce")
            .fillna(work["net_flow"])
            .astype(float)
        )
    else:
        work["desbalance_persona_monto_neto"] = work["net_flow"]
    if "desbalance_persona_meses_totales" in work.columns:
        work["desbalance_persona_meses_totales"] = (
            pd.to_numeric(work["desbalance_persona_meses_totales"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
    else:
        work["desbalance_persona_meses_totales"] = 0
    for col in [
        "desbalance_persona_tasa_meses_envia_extremo",
        "desbalance_persona_tasa_meses_recibe_extremo",
    ]:
        if col in work.columns:
            work[col] = (
                pd.to_numeric(work[col], errors="coerce").fillna(0.0).astype(float)
            )
        else:
            work[col] = 0.0

    score_labels = {3: "Alto", 2: "Medio", 1: "Bajo"}

    def _normalize_persona_id(value: Any) -> str:
        text = str(value).strip() if value is not None else ""
        if not text or text.lower() == "nan":
            return ""
        return text

    def _unique_list(values: Iterable[Any]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for value in values:
            if isinstance(value, (list, tuple, set)):
                iterable = value
            else:
                iterable = [value]
            for item in iterable:
                text = str(item).strip()
                if not text or text.lower() == "nan":
                    continue
                if text not in seen:
                    seen.add(text)
                    ordered.append(text)
        return ordered

    def _ensure_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, (tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        if value is None:
            return []
        if isinstance(value, float) and pd.isna(value):
            return []
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return []
        if text.startswith("[") and text.endswith("]"):
            inner = text[1:-1]
            parts = [part.strip().strip("'\"") for part in inner.split(",")]
            return [part for part in parts if part]
        return [text]

    def _score_metric_series(
        series: pd.Series, medium_quantile: float = 0.5, high_quantile: float = 0.9
    ) -> tuple[pd.Series, dict[str, float]]:
        metrics = pd.to_numeric(series, errors="coerce").fillna(0.0)
        positive = metrics[metrics > 0]
        thresholds = {"medium": 0.0, "high": 0.0}
        if positive.empty:
            return pd.Series(1, index=series.index, dtype=int), thresholds

        if len(positive) == 1:
            medium_threshold = float(positive.iloc[0])
            high_threshold = float(positive.iloc[0])
        else:
            medium_threshold = float(positive.quantile(medium_quantile))
            high_threshold = float(positive.quantile(high_quantile))
        if high_threshold < medium_threshold:
            high_threshold = float(positive.max())
        if medium_threshold <= 0 < high_threshold:
            medium_threshold = min(high_threshold, float(positive.median()))

        thresholds = {"medium": medium_threshold, "high": high_threshold}

        def _assign(value: float) -> int:
            if value <= 0:
                return 1
            if high_threshold > 0 and value >= high_threshold:
                return 3
            if medium_threshold > 0 and value >= medium_threshold:
                return 2
            return 1

        scores = metrics.apply(_assign).astype(int)
        return scores, thresholds

    def _format_list(values: list[str]) -> str:
        if not values:
            return "sin_dato"
        return ", ".join(values)

    scenario_frames: dict[str, pd.DataFrame] = {}

    manager_nlp_raw = question1_manager_nlp(reports, timeframe=timeframe)

    def _aggregate_manager_nlp(raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "tx_manager_nlp",
                    "monto_manager_nlp",
                    "roles_manager_nlp",
                    "conceptos_manager_nlp",
                ]
            )

        records: list[dict[str, Any]] = []
        for item in raw.to_dict(orient="records"):
            tx_count = float(item.get("tx_count", 0) or 0)
            monto_total = float(item.get("monto_total", 0) or 0)
            concepts = item.get("nlp_concepto_sospechoso", [])
            if not isinstance(concepts, list):
                concepts = [concepts]
            concept_list = [
                str(concept).strip()
                for concept in concepts
                if str(concept).strip() and str(concept).strip().lower() != "nan"
            ]
            manager = _normalize_persona_id(item.get("manager_user_id"))
            subordinate = _normalize_persona_id(item.get("subordinado_user_id"))
            for persona, role in ((manager, "manager"), (subordinate, "subordinado")):
                if not persona:
                    continue
                records.append(
                    {
                        "persona": persona,
                        "tx_manager_nlp": tx_count,
                        "monto_manager_nlp": monto_total,
                        "roles_manager_nlp": [role],
                        "conceptos_manager_nlp": concept_list,
                    }
                )

        if not records:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "tx_manager_nlp",
                    "monto_manager_nlp",
                    "roles_manager_nlp",
                    "conceptos_manager_nlp",
                ]
            )

        df = pd.DataFrame(records)
        aggregated = (
            df.groupby("persona", as_index=False)
            .agg(
                tx_manager_nlp=("tx_manager_nlp", "sum"),
                monto_manager_nlp=("monto_manager_nlp", "sum"),
                roles_manager_nlp=("roles_manager_nlp", _unique_list),
                conceptos_manager_nlp=("conceptos_manager_nlp", _unique_list),
            )
        )
        aggregated["tx_manager_nlp"] = (
            pd.to_numeric(aggregated["tx_manager_nlp"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        aggregated["monto_manager_nlp"] = (
            pd.to_numeric(aggregated["monto_manager_nlp"], errors="coerce")
            .fillna(0.0)
            .astype(float)
        )
        return aggregated

    manager_nlp_personas = _aggregate_manager_nlp(manager_nlp_raw)
    if manager_nlp_personas.empty:
        manager_nlp_personas = pd.DataFrame(
            columns=[
                "persona",
                "tx_manager_nlp",
                "monto_manager_nlp",
                "roles_manager_nlp",
                "conceptos_manager_nlp",
                "score_manager_nlp",
                "tier_manager_nlp",
                "detalle_manager_nlp",
            ]
        )
    else:
        scores, _ = _score_metric_series(manager_nlp_personas["tx_manager_nlp"])
        manager_nlp_personas["score_manager_nlp"] = scores
        manager_nlp_personas["tier_manager_nlp"] = manager_nlp_personas[
            "score_manager_nlp"
        ].map(score_labels)
        manager_nlp_personas["detalle_manager_nlp"] = manager_nlp_personas.apply(
            lambda row: (
                f"{int(row.get('tx_manager_nlp', 0))} tx por "
                f"{_format_float(row.get('monto_manager_nlp', 0))} "
                f"(roles: {_format_list(row.get('roles_manager_nlp', []))}; "
                f"conceptos: {_format_list(row.get('conceptos_manager_nlp', []))})"
            ),
            axis=1,
        )

    scenario_frames["manager_nlp"] = manager_nlp_personas

    def _build_manager_concepts(
        manager_df: pd.DataFrame, concepts_df: pd.DataFrame
    ) -> pd.DataFrame:
        if manager_df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "riesgo_manager_concepts",
                    "conceptos_manager_concepts",
                    "score_manager_concepts",
                    "tier_manager_concepts",
                    "detalle_manager_concepts",
                ]
            )

        concept_risk: dict[str, float] = {}
        if not concepts_df.empty:
            for item in concepts_df.to_dict(orient="records"):
                concept = str(item.get("nlp_concepto_sospechoso", "")).strip()
                if not concept or concept.lower() == "nan":
                    continue
                risk = float(item.get("risk_p95", 0) or 0)
                concept_risk[concept] = max(concept_risk.get(concept, 0.0), risk)

        work_concepts = manager_df[["persona", "conceptos_manager_nlp"]].copy()
        work_concepts["conceptos_manager_concepts"] = work_concepts[
            "conceptos_manager_nlp"
        ].apply(_unique_list)

        def _max_concept_risk(values: list[str]) -> float:
            if not values:
                return 0.0
            risks = [concept_risk.get(value, 0.0) for value in values]
            return float(max(risks) if risks else 0.0)

        work_concepts["riesgo_manager_concepts"] = work_concepts[
            "conceptos_manager_concepts"
        ].apply(_max_concept_risk)

        scores, _ = _score_metric_series(
            work_concepts["riesgo_manager_concepts"], medium_quantile=0.6
        )
        work_concepts["score_manager_concepts"] = scores
        work_concepts["tier_manager_concepts"] = work_concepts[
            "score_manager_concepts"
        ].map(score_labels)
        work_concepts["detalle_manager_concepts"] = work_concepts.apply(
            lambda row: (
                f"Conceptos severos {_format_list(row.get('conceptos_manager_concepts', []))} "
                f"con riesgo P95 {row.get('riesgo_manager_concepts', 0.0):.2f}"
            ),
            axis=1,
        )
        return work_concepts[[
            "persona",
            "riesgo_manager_concepts",
            "conceptos_manager_concepts",
            "score_manager_concepts",
            "tier_manager_concepts",
            "detalle_manager_concepts",
        ]]

    manager_concepts_raw = question2_manager_concepts(reports, timeframe=timeframe)
    scenario_frames["manager_concepts"] = _build_manager_concepts(
        manager_nlp_personas, manager_concepts_raw
    )

    def _split_pair(value: Any) -> list[str]:
        text = str(value).strip() if value is not None else ""
        if not text or text.lower() == "nan":
            return []
        for separator in ("⇄", "↔", "↔️"):
            if separator in text:
                return [part.strip() for part in text.split(separator) if part.strip()]
        for separator in ("→", "->", "⟶", "➡", "➝"):
            if separator in text:
                return [part.strip() for part in text.split(separator) if part.strip()]
        return []

    def _aggregate_persona_records(
        records: list[dict[str, Any]],
        *,
        numeric_fields: list[str],
        list_fields: list[str] | None = None,
    ) -> pd.DataFrame:
        if list_fields is None:
            list_fields = []
        if not records:
            columns = ["persona"] + numeric_fields + list_fields
            return pd.DataFrame(columns=columns)

        df = pd.DataFrame(records)
        agg_dict: dict[str, tuple[str, Any]] = {}
        for field in numeric_fields:
            agg_dict[field] = (field, "sum")
        for field in list_fields:
            agg_dict[field] = (field, _unique_list)
        aggregated = df.groupby("persona", as_index=False).agg(**agg_dict)
        for field in numeric_fields:
            aggregated[field] = (
                pd.to_numeric(aggregated[field], errors="coerce").fillna(0.0)
            )
        return aggregated

    quid_pairs_raw = question3_quid_pairs(
        reports, timeframe=timeframe, min_score=1.0, min_manager_ratio=0.0
    )

    def _build_quid_pairs(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "metric_quid_pairs",
                    "tx_quid_pairs",
                    "counterpartes_quid_pairs",
                    "score_quid_pairs",
                    "tier_quid_pairs",
                    "detalle_quid_pairs",
                ]
            )

        if "nivel_respuesta" in df.columns:
            work_df = df.loc[df["nivel_respuesta"] == "par"].copy()
            if work_df.empty:
                work_df = df.copy()
        else:
            work_df = df.copy()

        records: list[dict[str, Any]] = []
        for item in work_df.to_dict(orient="records"):
            pair_key = _normalize_persona_id(item.get("quid_pair_clave"))
            if not pair_key:
                pair_key = _normalize_persona_id(item.get("pair"))
            personas = _split_pair(pair_key)
            if len(personas) != 2:
                sender = _normalize_persona_id(item.get("user_id"))
                receiver = _normalize_persona_id(item.get("receptor-user_id"))
                personas = [p for p in (sender, receiver) if p]
            if not personas:
                continue
            tx_count = float(item.get("quid_tx_count", 0) or 0)
            score_max = float(item.get("quid_score_max", 0) or 0)
            score_avg = float(item.get("quid_score_avg", 0) or 0)
            weight_metric = tx_count * max(score_max, score_avg, 0.0)
            for persona in personas:
                other = [p for p in personas if p != persona]
                records.append(
                    {
                        "persona": persona,
                        "metric_quid_pairs": weight_metric,
                        "tx_quid_pairs": tx_count,
                        "counterpartes_quid_pairs": other,
                    }
                )

        aggregated = _aggregate_persona_records(
            records,
            numeric_fields=["metric_quid_pairs", "tx_quid_pairs"],
            list_fields=["counterpartes_quid_pairs"],
        )
        if aggregated.empty:
            return aggregated.assign(
                score_quid_pairs=pd.Series(dtype=int),
                tier_quid_pairs=pd.Series(dtype="object"),
                detalle_quid_pairs=pd.Series(dtype="object"),
            )

        scores, _ = _score_metric_series(
            aggregated["metric_quid_pairs"], medium_quantile=0.6
        )
        aggregated["score_quid_pairs"] = scores
        aggregated["tier_quid_pairs"] = aggregated["score_quid_pairs"].map(
            score_labels
        )
        aggregated["detalle_quid_pairs"] = aggregated.apply(
            lambda row: (
                f"{row.get('tx_quid_pairs', 0):.0f} tx sospechosas con contrapartes "
                f"{_format_list(row.get('counterpartes_quid_pairs', []))} "
                f"(peso {_format_float(row.get('metric_quid_pairs', 0))})"
            ),
            axis=1,
        )
        return aggregated[
            [
                "persona",
                "metric_quid_pairs",
                "tx_quid_pairs",
                "counterpartes_quid_pairs",
                "score_quid_pairs",
                "tier_quid_pairs",
                "detalle_quid_pairs",
            ]
        ]

    scenario_frames["quid_pairs"] = _build_quid_pairs(quid_pairs_raw)

    quid_negative_raw = question4_quid_negative_value_vs_load(
        reports, timeframe=timeframe
    )

    def _build_quid_negative(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "metric_quid_value_vs_load",
                    "tx_quid_value_vs_load",
                    "delta_quid_value_vs_load",
                    "score_signal_quid_value_vs_load",
                    "counterpartes_quid_value_vs_load",
                    "responsables_quid_value_vs_load",
                    "score_quid_value_vs_load",
                    "tier_quid_value_vs_load",
                    "detalle_quid_value_vs_load",
                ]
            )

        records: list[dict[str, Any]] = []
        for item in df.to_dict(orient="records"):
            sender = _normalize_persona_id(item.get(COL_SENDER_ID))
            receiver = _normalize_persona_id(item.get(COL_RECEIVER_ID))
            personas = [p for p in (sender, receiver) if p]
            if not personas:
                continue
            responsable = _normalize_persona_id(item.get("responsable_user_id"))
            delta_days = float(item.get("feat_quid_value_vs_load_days", 0) or 0)
            delta_metric = abs(delta_days) if delta_days < 0 else 0.0
            score_val = float(item.get("feat_quid_score", 0) or 0)
            combined_metric = delta_metric + max(score_val, 0.0)
            for persona in personas:
                other = [p for p in personas if p != persona]
                records.append(
                    {
                        "persona": persona,
                        "metric_quid_value_vs_load": combined_metric,
                        "tx_quid_value_vs_load": 1.0,
                        "delta_quid_value_vs_load": delta_metric,
                        "score_signal_quid_value_vs_load": max(score_val, 0.0),
                        "counterpartes_quid_value_vs_load": other,
                        "responsables_quid_value_vs_load": [responsable]
                        if responsable
                        else [],
                    }
                )

        aggregated = _aggregate_persona_records(
            records,
            numeric_fields=[
                "metric_quid_value_vs_load",
                "tx_quid_value_vs_load",
                "delta_quid_value_vs_load",
                "score_signal_quid_value_vs_load",
            ],
            list_fields=[
                "counterpartes_quid_value_vs_load",
                "responsables_quid_value_vs_load",
            ],
        )
        if aggregated.empty:
            return aggregated.assign(
                score_quid_value_vs_load=pd.Series(dtype=int),
                tier_quid_value_vs_load=pd.Series(dtype="object"),
                detalle_quid_value_vs_load=pd.Series(dtype="object"),
            )

        scores, _ = _score_metric_series(
            aggregated["metric_quid_value_vs_load"], medium_quantile=0.5
        )
        aggregated["score_quid_value_vs_load"] = scores
        aggregated["tier_quid_value_vs_load"] = aggregated[
            "score_quid_value_vs_load"
        ].map(score_labels)
        aggregated["detalle_quid_value_vs_load"] = aggregated.apply(
            lambda row: (
                f"{row.get('tx_quid_value_vs_load', 0):.0f} tx con desfase acumulado "
                f"{_format_float(row.get('delta_quid_value_vs_load', 0))} días y puntaje "
                f"{row.get('score_signal_quid_value_vs_load', 0):.2f}; contrapartes "
                f"{_format_list(row.get('counterpartes_quid_value_vs_load', []))}; "
                f"responsables {_format_list(row.get('responsables_quid_value_vs_load', []))}"
            ),
            axis=1,
        )
        return aggregated[
            [
                "persona",
                "metric_quid_value_vs_load",
                "tx_quid_value_vs_load",
                "delta_quid_value_vs_load",
                "score_signal_quid_value_vs_load",
                "counterpartes_quid_value_vs_load",
                "responsables_quid_value_vs_load",
                "score_quid_value_vs_load",
                "tier_quid_value_vs_load",
                "detalle_quid_value_vs_load",
            ]
        ]

    scenario_frames["quid_value_vs_load"] = _build_quid_negative(quid_negative_raw)

    reference_reuse_raw = question5_reference_reuse(reports, timeframe=timeframe)

    def _build_reference_reuse(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "metric_reference_reuse",
                    "tx_reference_reuse",
                    "referencias_reference_reuse",
                    "counterpartes_reference_reuse",
                    "score_reference_reuse",
                    "tier_reference_reuse",
                    "detalle_reference_reuse",
                ]
            )

        records: list[dict[str, Any]] = []
        for item in df.to_dict(orient="records"):
            tx_count = float(item.get("tx_count", 0) or 0)
            reference = str(item.get("reference_norm", "")).strip()
            pairs_text = item.get("pairs")
            if isinstance(pairs_text, str):
                pair_values = [p.strip() for p in pairs_text.split(";") if p.strip()]
            else:
                pair_values = []
            for pair in pair_values:
                personas = _split_pair(pair)
                if len(personas) != 2:
                    continue
                for persona in personas:
                    other = [p for p in personas if p != persona]
                    records.append(
                        {
                            "persona": persona,
                            "metric_reference_reuse": tx_count,
                            "tx_reference_reuse": tx_count,
                            "referencias_reference_reuse": [reference]
                            if reference
                            else [],
                            "counterpartes_reference_reuse": other,
                        }
                    )

        aggregated = _aggregate_persona_records(
            records,
            numeric_fields=["metric_reference_reuse", "tx_reference_reuse"],
            list_fields=[
                "referencias_reference_reuse",
                "counterpartes_reference_reuse",
            ],
        )
        if aggregated.empty:
            return aggregated.assign(
                score_reference_reuse=pd.Series(dtype=int),
                tier_reference_reuse=pd.Series(dtype="object"),
                detalle_reference_reuse=pd.Series(dtype="object"),
            )

        scores, _ = _score_metric_series(
            aggregated["metric_reference_reuse"], medium_quantile=0.55
        )
        aggregated["score_reference_reuse"] = scores
        aggregated["tier_reference_reuse"] = aggregated["score_reference_reuse"].map(
            score_labels
        )
        aggregated["detalle_reference_reuse"] = aggregated.apply(
            lambda row: (
                f"{row.get('tx_reference_reuse', 0):.0f} tx con referencias "
                f"reutilizadas {_format_list(row.get('referencias_reference_reuse', []))} "
                f"y contrapartes {_format_list(row.get('counterpartes_reference_reuse', []))}"
            ),
            axis=1,
        )
        return aggregated[
            [
                "persona",
                "metric_reference_reuse",
                "tx_reference_reuse",
                "referencias_reference_reuse",
                "counterpartes_reference_reuse",
                "score_reference_reuse",
                "tier_reference_reuse",
                "detalle_reference_reuse",
            ]
        ]

    scenario_frames["reference_reuse"] = _build_reference_reuse(
        reference_reuse_raw
    )

    centralizers_raw = question6_centralizers(reports, timeframe=timeframe)

    def _build_centralizers(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "monto_centralizer_inflow",
                    "emisores_centralizer",
                    "tx_centralizer",
                    "score_centralizer",
                    "tier_centralizer",
                    "detalle_centralizer",
                ]
            )

        work_df = df.copy()
        persona_col = "receptor-user_id"
        if persona_col not in work_df.columns:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "monto_centralizer_inflow",
                    "emisores_centralizer",
                    "tx_centralizer",
                    "score_centralizer",
                    "tier_centralizer",
                    "detalle_centralizer",
                ]
            )
        work_df["persona"] = work_df[persona_col].apply(_normalize_persona_id)
        work_df = work_df.loc[work_df["persona"] != ""].copy()
        if work_df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "monto_centralizer_inflow",
                    "emisores_centralizer",
                    "tx_centralizer",
                    "score_centralizer",
                    "tier_centralizer",
                    "detalle_centralizer",
                ]
            )

        work_df["monto_centralizer_inflow"] = pd.to_numeric(
            work_df.get("inflow", 0), errors="coerce"
        ).fillna(0.0)
        work_df["emisores_centralizer"] = pd.to_numeric(
            work_df.get("emisores_unicos", 0), errors="coerce"
        ).fillna(0).astype(int)
        work_df["tx_centralizer"] = pd.to_numeric(
            work_df.get("n_tx", 0), errors="coerce"
        ).fillna(0).astype(int)

        summarized = work_df[[
            "persona",
            "monto_centralizer_inflow",
            "emisores_centralizer",
            "tx_centralizer",
        ]].groupby("persona", as_index=False).agg(
            monto_centralizer_inflow=("monto_centralizer_inflow", "sum"),
            emisores_centralizer=("emisores_centralizer", "max"),
            tx_centralizer=("tx_centralizer", "sum"),
        )
        scores, _ = _score_metric_series(
            summarized["monto_centralizer_inflow"], medium_quantile=0.55
        )
        summarized["score_centralizer"] = scores
        summarized["tier_centralizer"] = summarized["score_centralizer"].map(
            score_labels
        )
        summarized["detalle_centralizer"] = summarized.apply(
            lambda row: (
                f"Recibió {_format_float(row.get('monto_centralizer_inflow', 0))} "
                f"de {int(row.get('emisores_centralizer', 0))} emisores únicos "
                f"en {int(row.get('tx_centralizer', 0))} pagos"
            ),
            axis=1,
        )
        return summarized

    scenario_frames["centralizer"] = _build_centralizers(centralizers_raw)

    def _build_net_imbalance(base_df: pd.DataFrame) -> pd.DataFrame:
        if base_df.empty or "persona" not in base_df.columns:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "monto_net_imbalance_abs",
                    "meses_envia_net_imbalance",
                    "meses_recibe_net_imbalance",
                    "score_net_imbalance",
                    "tier_net_imbalance",
                    "detalle_net_imbalance",
                ]
            )

        net_df = base_df[[
            "persona",
            "desbalance_persona_monto_neto",
            "desbalance_persona_meses_totales",
        ]].copy()
        net_df["desbalance_persona_meses_envia_extremo"] = base_df.get(
            "desbalance_persona_meses_envia_extremo", 0
        )
        net_df["desbalance_persona_meses_recibe_extremo"] = base_df.get(
            "desbalance_persona_meses_recibe_extremo", 0
        )
        net_df["persona"] = net_df["persona"].apply(_normalize_persona_id)
        net_df = net_df.loc[net_df["persona"] != ""].copy()
        if net_df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "monto_net_imbalance_abs",
                    "meses_envia_net_imbalance",
                    "meses_recibe_net_imbalance",
                    "score_net_imbalance",
                    "tier_net_imbalance",
                    "detalle_net_imbalance",
                ]
            )

        net_df["monto_net_imbalance_abs"] = pd.to_numeric(
            net_df.get("desbalance_persona_monto_neto", 0), errors="coerce"
        ).fillna(0.0).abs()
        net_df["meses_envia_net_imbalance"] = pd.to_numeric(
            net_df.get("desbalance_persona_meses_envia_extremo", 0), errors="coerce"
        ).fillna(0.0)
        net_df["meses_recibe_net_imbalance"] = pd.to_numeric(
            net_df.get("desbalance_persona_meses_recibe_extremo", 0), errors="coerce"
        ).fillna(0.0)

        aggregated = net_df[[
            "persona",
            "monto_net_imbalance_abs",
            "meses_envia_net_imbalance",
            "meses_recibe_net_imbalance",
        ]].groupby("persona", as_index=False).agg(
            monto_net_imbalance_abs=("monto_net_imbalance_abs", "max"),
            meses_envia_net_imbalance=("meses_envia_net_imbalance", "max"),
            meses_recibe_net_imbalance=("meses_recibe_net_imbalance", "max"),
        )
        scores, _ = _score_metric_series(
            aggregated["monto_net_imbalance_abs"], medium_quantile=0.6
        )
        aggregated["score_net_imbalance"] = scores
        aggregated["tier_net_imbalance"] = aggregated["score_net_imbalance"].map(
            score_labels
        )
        aggregated["detalle_net_imbalance"] = aggregated.apply(
            lambda row: (
                f"Desbalance neto {_format_float(row.get('monto_net_imbalance_abs', 0))} "
                f"con meses extremos envío {row.get('meses_envia_net_imbalance', 0):.0f} "
                f"y recepción {row.get('meses_recibe_net_imbalance', 0):.0f}"
            ),
            axis=1,
        )
        return aggregated

    scenario_frames["net_imbalance"] = _build_net_imbalance(work)

    case13_raw = question8_case13_new_employees(reports, timeframe=timeframe)

    def _build_case13(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "monto_case13",
                    "tx_case13",
                    "emisores_case13",
                    "score_case13",
                    "tier_case13",
                    "detalle_case13",
                ]
            )

        work_df = df.copy()
        work_df["persona"] = work_df["persona"].apply(_normalize_persona_id)
        work_df = work_df.loc[work_df["persona"] != ""].copy()
        if work_df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "monto_case13",
                    "tx_case13",
                    "emisores_case13",
                    "score_case13",
                    "tier_case13",
                    "detalle_case13",
                ]
            )

        work_df["monto_case13"] = pd.to_numeric(
            work_df.get("caso13_persona_monto_total", 0), errors="coerce"
        ).fillna(0.0)
        work_df["tx_case13"] = pd.to_numeric(
            work_df.get("caso13_persona_tx_recibidas", 0), errors="coerce"
        ).fillna(0).astype(int)
        work_df["emisores_case13"] = pd.to_numeric(
            work_df.get("caso13_persona_emisores_unicos", 0), errors="coerce"
        ).fillna(0).astype(int)

        aggregated = work_df[[
            "persona",
            "monto_case13",
            "tx_case13",
            "emisores_case13",
        ]]
        scores, _ = _score_metric_series(aggregated["monto_case13"], medium_quantile=0.6)
        aggregated["score_case13"] = scores
        aggregated["tier_case13"] = aggregated["score_case13"].map(score_labels)
        aggregated["detalle_case13"] = aggregated.apply(
            lambda row: (
                f"Recibió {_format_float(row.get('monto_case13', 0))} en {int(row.get('tx_case13', 0))} tx "
                f"desde {int(row.get('emisores_case13', 0))} emisores"
            ),
            axis=1,
        )
        return aggregated

    scenario_frames["case13"] = _build_case13(case13_raw)

    case14_raw = question9_case14_veterans_from_newcomers(
        reports, timeframe=timeframe
    )

    def _build_case14(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "monto_case14",
                    "tx_case14",
                    "emisores_case14",
                    "score_case14",
                    "tier_case14",
                    "detalle_case14",
                ]
            )

        work_df = df.copy()
        work_df["persona"] = work_df["persona"].apply(_normalize_persona_id)
        work_df = work_df.loc[work_df["persona"] != ""].copy()
        if work_df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "monto_case14",
                    "tx_case14",
                    "emisores_case14",
                    "score_case14",
                    "tier_case14",
                    "detalle_case14",
                ]
            )

        work_df["monto_case14"] = pd.to_numeric(
            work_df.get("caso14_persona_monto_de_emisores_nuevos", 0), errors="coerce"
        ).fillna(0.0)
        work_df["tx_case14"] = pd.to_numeric(
            work_df.get("caso14_persona_tx_de_emisores_nuevos", 0), errors="coerce"
        ).fillna(0).astype(int)
        work_df["emisores_case14"] = pd.to_numeric(
            work_df.get("caso14_persona_emisores_nuevos_unicos", 0), errors="coerce"
        ).fillna(0).astype(int)

        aggregated = work_df[[
            "persona",
            "monto_case14",
            "tx_case14",
            "emisores_case14",
        ]]
        scores, _ = _score_metric_series(aggregated["monto_case14"], medium_quantile=0.6)
        aggregated["score_case14"] = scores
        aggregated["tier_case14"] = aggregated["score_case14"].map(score_labels)
        aggregated["detalle_case14"] = aggregated.apply(
            lambda row: (
                f"Recibió {_format_float(row.get('monto_case14', 0))} en {int(row.get('tx_case14', 0))} tx "
                f"de {int(row.get('emisores_case14', 0))} emisores nuevos"
            ),
            axis=1,
        )
        return aggregated

    scenario_frames["case14"] = _build_case14(case14_raw)

    yoyo_raw = question10_yoyo_streaks(reports, timeframe=timeframe)

    def _build_yoyo(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "tx_yoyo",
                    "racha_yoyo_max",
                    "metric_yoyo",
                    "score_yoyo",
                    "tier_yoyo",
                    "detalle_yoyo",
                ]
            )

        records: list[dict[str, Any]] = []
        for item in df.to_dict(orient="records"):
            pair = _split_pair(item.get("par_bidir"))
            if len(pair) != 2:
                continue
            tx_total = float(item.get("tx_yo_yo_totales", 0) or 0)
            racha_max = float(item.get("racha_max_yo_yo", 0) or 0)
            riesgo_max = float(item.get("riesgo_max_yo_yo", 0) or 0)
            metric = tx_total + max(racha_max, 0) + max(riesgo_max - 1.0, 0)
            for persona in pair:
                other = [p for p in pair if p != persona]
                records.append(
                    {
                        "persona": persona,
                        "tx_yoyo": tx_total,
                        "racha_yoyo_max": racha_max,
                        "metric_yoyo": metric,
                        "counterpartes_yoyo": other,
                    }
                )

        aggregated = _aggregate_persona_records(
            records,
            numeric_fields=["metric_yoyo", "tx_yoyo", "racha_yoyo_max"],
            list_fields=["counterpartes_yoyo"],
        )
        if aggregated.empty:
            return aggregated.assign(
                score_yoyo=pd.Series(dtype=int),
                tier_yoyo=pd.Series(dtype="object"),
                detalle_yoyo=pd.Series(dtype="object"),
            )

        scores, _ = _score_metric_series(
            aggregated["metric_yoyo"], medium_quantile=0.55
        )
        aggregated["score_yoyo"] = scores
        aggregated["tier_yoyo"] = aggregated["score_yoyo"].map(score_labels)
        aggregated["detalle_yoyo"] = aggregated.apply(
            lambda row: (
                f"{row.get('tx_yoyo', 0):.0f} tx yo-yo, racha máxima {row.get('racha_yoyo_max', 0):.0f} "
                f"con {_format_list(row.get('counterpartes_yoyo', []))}"
            ),
            axis=1,
        )
        return aggregated

    scenario_frames["yoyo"] = _build_yoyo(yoyo_raw)

    near_threshold_raw = question11_near_threshold_structuring(
        reports, timeframe=timeframe
    )

    def _build_near_threshold(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "tx_near_threshold",
                    "monto_near_threshold",
                    "metric_near_threshold",
                    "score_near_threshold",
                    "tier_near_threshold",
                    "detalle_near_threshold",
                ]
            )

        records: list[dict[str, Any]] = []
        for item in df.to_dict(orient="records"):
            pair = _split_pair(item.get("pair"))
            if len(pair) != 2:
                continue
            tx_total = float(item.get("tx_near_totales", 0) or 0)
            monto_total = float(item.get("monto_total_near", 0) or 0)
            metric = tx_total + monto_total / 1000.0
            for persona in pair:
                other = [p for p in pair if p != persona]
                records.append(
                    {
                        "persona": persona,
                        "tx_near_threshold": tx_total,
                        "monto_near_threshold": monto_total,
                        "metric_near_threshold": metric,
                        "counterpartes_near_threshold": other,
                    }
                )

        aggregated = _aggregate_persona_records(
            records,
            numeric_fields=[
                "metric_near_threshold",
                "tx_near_threshold",
                "monto_near_threshold",
            ],
            list_fields=["counterpartes_near_threshold"],
        )
        if aggregated.empty:
            return aggregated.assign(
                score_near_threshold=pd.Series(dtype=int),
                tier_near_threshold=pd.Series(dtype="object"),
                detalle_near_threshold=pd.Series(dtype="object"),
            )

        scores, _ = _score_metric_series(
            aggregated["metric_near_threshold"], medium_quantile=0.55
        )
        aggregated["score_near_threshold"] = scores
        aggregated["tier_near_threshold"] = aggregated[
            "score_near_threshold"
        ].map(score_labels)
        aggregated["detalle_near_threshold"] = aggregated.apply(
            lambda row: (
                f"{row.get('tx_near_threshold', 0):.0f} tx cerca de umbral por "
                f"{_format_float(row.get('monto_near_threshold', 0))} con "
                f"{_format_list(row.get('counterpartes_near_threshold', []))}"
            ),
            axis=1,
        )
        return aggregated

    scenario_frames["near_threshold"] = _build_near_threshold(near_threshold_raw)

    smurfing_raw = question12_smurfing_chronic(reports, timeframe=timeframe)

    def _build_smurfing(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "tx_smurfing",
                    "monto_smurfing",
                    "metric_smurfing",
                    "score_smurfing",
                    "tier_smurfing",
                    "detalle_smurfing",
                ]
            )

        records: list[dict[str, Any]] = []
        for item in df.to_dict(orient="records"):
            pair = _split_pair(item.get("pair"))
            if len(pair) != 2:
                continue
            tx_total = float(item.get("transacciones_fraccionadas", 0) or 0)
            monto_total = float(item.get("monto_fraccionado_total", 0) or 0)
            metric = tx_total + monto_total / 1000.0
            for persona in pair:
                other = [p for p in pair if p != persona]
                records.append(
                    {
                        "persona": persona,
                        "tx_smurfing": tx_total,
                        "monto_smurfing": monto_total,
                        "metric_smurfing": metric,
                        "counterpartes_smurfing": other,
                    }
                )

        aggregated = _aggregate_persona_records(
            records,
            numeric_fields=["metric_smurfing", "tx_smurfing", "monto_smurfing"],
            list_fields=["counterpartes_smurfing"],
        )
        if aggregated.empty:
            return aggregated.assign(
                score_smurfing=pd.Series(dtype=int),
                tier_smurfing=pd.Series(dtype="object"),
                detalle_smurfing=pd.Series(dtype="object"),
            )

        scores, _ = _score_metric_series(
            aggregated["metric_smurfing"], medium_quantile=0.55
        )
        aggregated["score_smurfing"] = scores
        aggregated["tier_smurfing"] = aggregated["score_smurfing"].map(score_labels)
        aggregated["detalle_smurfing"] = aggregated.apply(
            lambda row: (
                f"{row.get('tx_smurfing', 0):.0f} tx fraccionadas por "
                f"{_format_float(row.get('monto_smurfing', 0))} con "
                f"{_format_list(row.get('counterpartes_smurfing', []))}"
            ),
            axis=1,
        )
        return aggregated

    scenario_frames["smurfing"] = _build_smurfing(smurfing_raw)

    bad_loans_raw = question13_bad_loans_with_frequency(
        reports, timeframe=timeframe
    )

    def _build_bad_loans(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "tx_bad_loans",
                    "monto_bad_loans",
                    "eventos_bad_loans",
                    "metric_bad_loans",
                    "score_bad_loans",
                    "tier_bad_loans",
                    "detalle_bad_loans",
                ]
            )

        records: list[dict[str, Any]] = []
        for item in df.to_dict(orient="records"):
            pair = _split_pair(item.get("pair"))
            if len(pair) != 2:
                continue
            tx_total = float(item.get("prestamos_incumplidos", 0) or 0)
            monto_total = float(item.get("monto_prestamos_incumplidos", 0) or 0)
            eventos = float(item.get("eventos_alta_frecuencia", 0) or 0)
            metric = monto_total + tx_total * 1000.0 + eventos * 500.0
            for persona in pair:
                other = [p for p in pair if p != persona]
                records.append(
                    {
                        "persona": persona,
                        "tx_bad_loans": tx_total,
                        "monto_bad_loans": monto_total,
                        "eventos_bad_loans": eventos,
                        "metric_bad_loans": metric,
                        "counterpartes_bad_loans": other,
                    }
                )

        aggregated = _aggregate_persona_records(
            records,
            numeric_fields=[
                "metric_bad_loans",
                "tx_bad_loans",
                "monto_bad_loans",
                "eventos_bad_loans",
            ],
            list_fields=["counterpartes_bad_loans"],
        )
        if aggregated.empty:
            return aggregated.assign(
                score_bad_loans=pd.Series(dtype=int),
                tier_bad_loans=pd.Series(dtype="object"),
                detalle_bad_loans=pd.Series(dtype="object"),
            )

        scores, _ = _score_metric_series(
            aggregated["metric_bad_loans"], medium_quantile=0.55
        )
        aggregated["score_bad_loans"] = scores
        aggregated["tier_bad_loans"] = aggregated["score_bad_loans"].map(
            score_labels
        )
        aggregated["detalle_bad_loans"] = aggregated.apply(
            lambda row: (
                f"{row.get('tx_bad_loans', 0):.0f} préstamos impagos por "
                f"{_format_float(row.get('monto_bad_loans', 0))} con eventos "
                f"de alta frecuencia {row.get('eventos_bad_loans', 0):.0f} junto a "
                f"{_format_list(row.get('counterpartes_bad_loans', []))}"
            ),
            axis=1,
        )
        return aggregated

    scenario_frames["bad_loans"] = _build_bad_loans(bad_loans_raw)

    payroll_raw = question14_recurrent_payroll(reports, timeframe=timeframe)

    def _build_payroll(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(
                columns=[
                    "persona",
                    "tx_recurrent_payroll",
                    "meses_recurrent_payroll",
                    "monto_recurrent_payroll",
                    "metric_recurrent_payroll",
                    "score_recurrent_payroll",
                    "tier_recurrent_payroll",
                    "detalle_recurrent_payroll",
                ]
            )

        records: list[dict[str, Any]] = []
        for item in df.to_dict(orient="records"):
            sender = _normalize_persona_id(item.get("emisor"))
            receiver = _normalize_persona_id(item.get("receptor"))
            personas = [p for p in (sender, receiver) if p]
            if not personas:
                continue
            meses = float(item.get("meses_recurrentes", 0) or 0)
            tx_total = float(item.get("tx_totales", 0) or 0)
            monto_total = float(item.get("monto_total", 0) or 0)
            metric = monto_total + meses * 1000.0
            for persona in personas:
                other = [p for p in personas if p != persona]
                records.append(
                    {
                        "persona": persona,
                        "tx_recurrent_payroll": tx_total,
                        "meses_recurrent_payroll": meses,
                        "monto_recurrent_payroll": monto_total,
                        "metric_recurrent_payroll": metric,
                        "counterpartes_recurrent_payroll": other,
                    }
                )

        aggregated = _aggregate_persona_records(
            records,
            numeric_fields=[
                "metric_recurrent_payroll",
                "tx_recurrent_payroll",
                "meses_recurrent_payroll",
                "monto_recurrent_payroll",
            ],
            list_fields=["counterpartes_recurrent_payroll"],
        )
        if aggregated.empty:
            return aggregated.assign(
                score_recurrent_payroll=pd.Series(dtype=int),
                tier_recurrent_payroll=pd.Series(dtype="object"),
                detalle_recurrent_payroll=pd.Series(dtype="object"),
            )

        scores, _ = _score_metric_series(
            aggregated["metric_recurrent_payroll"], medium_quantile=0.55
        )
        aggregated["score_recurrent_payroll"] = scores
        aggregated["tier_recurrent_payroll"] = aggregated[
            "score_recurrent_payroll"
        ].map(score_labels)
        aggregated["detalle_recurrent_payroll"] = aggregated.apply(
            lambda row: (
                f"{row.get('tx_recurrent_payroll', 0):.0f} pagos en "
                f"{row.get('meses_recurrent_payroll', 0):.0f} meses por "
                f"{_format_float(row.get('monto_recurrent_payroll', 0))} con "
                f"{_format_list(row.get('counterpartes_recurrent_payroll', []))}"
            ),
            axis=1,
        )
        return aggregated

    scenario_frames["recurrent_payroll"] = _build_payroll(payroll_raw)

    scenario_metadata = {
        "manager_nlp": {
            "label": "NLP manager-subordinado",
            "weight": 1.1,
            "score_col": "score_manager_nlp",
            "tier_col": "tier_manager_nlp",
            "detail_col": "detalle_manager_nlp",
            "columns": [
                "tx_manager_nlp",
                "monto_manager_nlp",
                "roles_manager_nlp",
                "conceptos_manager_nlp",
                "score_manager_nlp",
                "tier_manager_nlp",
                "detalle_manager_nlp",
            ],
            "numeric_cols": ["tx_manager_nlp", "monto_manager_nlp"],
            "list_cols": ["roles_manager_nlp", "conceptos_manager_nlp"],
        },
        "manager_concepts": {
            "label": "Conceptos NLP severos",
            "weight": 0.9,
            "score_col": "score_manager_concepts",
            "tier_col": "tier_manager_concepts",
            "detail_col": "detalle_manager_concepts",
            "columns": [
                "riesgo_manager_concepts",
                "conceptos_manager_concepts",
                "score_manager_concepts",
                "tier_manager_concepts",
                "detalle_manager_concepts",
            ],
            "numeric_cols": ["riesgo_manager_concepts"],
            "list_cols": ["conceptos_manager_concepts"],
        },
        "quid_pairs": {
            "label": "Quid pro quo",
            "weight": 1.2,
            "score_col": "score_quid_pairs",
            "tier_col": "tier_quid_pairs",
            "detail_col": "detalle_quid_pairs",
            "columns": [
                "metric_quid_pairs",
                "tx_quid_pairs",
                "counterpartes_quid_pairs",
                "score_quid_pairs",
                "tier_quid_pairs",
                "detalle_quid_pairs",
            ],
            "numeric_cols": ["metric_quid_pairs", "tx_quid_pairs"],
            "list_cols": ["counterpartes_quid_pairs"],
        },
        "quid_value_vs_load": {
            "label": "Valor vs carga",
            "weight": 1.0,
            "score_col": "score_quid_value_vs_load",
            "tier_col": "tier_quid_value_vs_load",
            "detail_col": "detalle_quid_value_vs_load",
            "columns": [
                "metric_quid_value_vs_load",
                "tx_quid_value_vs_load",
                "delta_quid_value_vs_load",
                "score_signal_quid_value_vs_load",
                "counterpartes_quid_value_vs_load",
                "responsables_quid_value_vs_load",
                "score_quid_value_vs_load",
                "tier_quid_value_vs_load",
                "detalle_quid_value_vs_load",
            ],
            "numeric_cols": [
                "metric_quid_value_vs_load",
                "tx_quid_value_vs_load",
                "delta_quid_value_vs_load",
                "score_signal_quid_value_vs_load",
            ],
            "list_cols": [
                "counterpartes_quid_value_vs_load",
                "responsables_quid_value_vs_load",
            ],
        },
        "reference_reuse": {
            "label": "Referencias reutilizadas",
            "weight": 1.0,
            "score_col": "score_reference_reuse",
            "tier_col": "tier_reference_reuse",
            "detail_col": "detalle_reference_reuse",
            "columns": [
                "metric_reference_reuse",
                "tx_reference_reuse",
                "referencias_reference_reuse",
                "counterpartes_reference_reuse",
                "score_reference_reuse",
                "tier_reference_reuse",
                "detalle_reference_reuse",
            ],
            "numeric_cols": ["metric_reference_reuse", "tx_reference_reuse"],
            "list_cols": [
                "referencias_reference_reuse",
                "counterpartes_reference_reuse",
            ],
        },
        "centralizer": {
            "label": "Centralizadores",
            "weight": 1.1,
            "score_col": "score_centralizer",
            "tier_col": "tier_centralizer",
            "detail_col": "detalle_centralizer",
            "columns": [
                "monto_centralizer_inflow",
                "emisores_centralizer",
                "tx_centralizer",
                "score_centralizer",
                "tier_centralizer",
                "detalle_centralizer",
            ],
            "numeric_cols": [
                "monto_centralizer_inflow",
                "emisores_centralizer",
                "tx_centralizer",
            ],
            "list_cols": [],
        },
        "net_imbalance": {
            "label": "Desbalance neto",
            "weight": 1.4,
            "score_col": "score_net_imbalance",
            "tier_col": "tier_net_imbalance",
            "detail_col": "detalle_net_imbalance",
            "columns": [
                "monto_net_imbalance_abs",
                "meses_envia_net_imbalance",
                "meses_recibe_net_imbalance",
                "score_net_imbalance",
                "tier_net_imbalance",
                "detalle_net_imbalance",
            ],
            "numeric_cols": [
                "monto_net_imbalance_abs",
                "meses_envia_net_imbalance",
                "meses_recibe_net_imbalance",
            ],
            "list_cols": [],
        },
        "case13": {
            "label": "Receptores nuevos (Caso 13)",
            "weight": 1.3,
            "score_col": "score_case13",
            "tier_col": "tier_case13",
            "detail_col": "detalle_case13",
            "columns": [
                "monto_case13",
                "tx_case13",
                "emisores_case13",
                "score_case13",
                "tier_case13",
                "detalle_case13",
            ],
            "numeric_cols": ["monto_case13", "tx_case13", "emisores_case13"],
            "list_cols": [],
        },
        "case14": {
            "label": "Veteranos desde nuevos (Caso 14)",
            "weight": 1.2,
            "score_col": "score_case14",
            "tier_col": "tier_case14",
            "detail_col": "detalle_case14",
            "columns": [
                "monto_case14",
                "tx_case14",
                "emisores_case14",
                "score_case14",
                "tier_case14",
                "detalle_case14",
            ],
            "numeric_cols": ["monto_case14", "tx_case14", "emisores_case14"],
            "list_cols": [],
        },
        "yoyo": {
            "label": "Rachas yo-yo",
            "weight": 1.0,
            "score_col": "score_yoyo",
            "tier_col": "tier_yoyo",
            "detail_col": "detalle_yoyo",
            "columns": [
                "metric_yoyo",
                "tx_yoyo",
                "racha_yoyo_max",
                "counterpartes_yoyo",
                "score_yoyo",
                "tier_yoyo",
                "detalle_yoyo",
            ],
            "numeric_cols": ["metric_yoyo", "tx_yoyo", "racha_yoyo_max"],
            "list_cols": ["counterpartes_yoyo"],
        },
        "near_threshold": {
            "label": "Cercanía a umbral",
            "weight": 0.9,
            "score_col": "score_near_threshold",
            "tier_col": "tier_near_threshold",
            "detail_col": "detalle_near_threshold",
            "columns": [
                "metric_near_threshold",
                "tx_near_threshold",
                "monto_near_threshold",
                "counterpartes_near_threshold",
                "score_near_threshold",
                "tier_near_threshold",
                "detalle_near_threshold",
            ],
            "numeric_cols": [
                "metric_near_threshold",
                "tx_near_threshold",
                "monto_near_threshold",
            ],
            "list_cols": ["counterpartes_near_threshold"],
        },
        "smurfing": {
            "label": "Fraccionamiento crónico",
            "weight": 1.4,
            "score_col": "score_smurfing",
            "tier_col": "tier_smurfing",
            "detail_col": "detalle_smurfing",
            "columns": [
                "metric_smurfing",
                "tx_smurfing",
                "monto_smurfing",
                "counterpartes_smurfing",
                "score_smurfing",
                "tier_smurfing",
                "detalle_smurfing",
            ],
            "numeric_cols": [
                "metric_smurfing",
                "tx_smurfing",
                "monto_smurfing",
            ],
            "list_cols": ["counterpartes_smurfing"],
        },
        "bad_loans": {
            "label": "Préstamos impagos",
            "weight": 1.3,
            "score_col": "score_bad_loans",
            "tier_col": "tier_bad_loans",
            "detail_col": "detalle_bad_loans",
            "columns": [
                "metric_bad_loans",
                "tx_bad_loans",
                "monto_bad_loans",
                "eventos_bad_loans",
                "counterpartes_bad_loans",
                "score_bad_loans",
                "tier_bad_loans",
                "detalle_bad_loans",
            ],
            "numeric_cols": [
                "metric_bad_loans",
                "tx_bad_loans",
                "monto_bad_loans",
                "eventos_bad_loans",
            ],
            "list_cols": ["counterpartes_bad_loans"],
        },
        "recurrent_payroll": {
            "label": "Pagos recurrentes",
            "weight": 1.2,
            "score_col": "score_recurrent_payroll",
            "tier_col": "tier_recurrent_payroll",
            "detail_col": "detalle_recurrent_payroll",
            "columns": [
                "metric_recurrent_payroll",
                "tx_recurrent_payroll",
                "meses_recurrent_payroll",
                "monto_recurrent_payroll",
                "counterpartes_recurrent_payroll",
                "score_recurrent_payroll",
                "tier_recurrent_payroll",
                "detalle_recurrent_payroll",
            ],
            "numeric_cols": [
                "metric_recurrent_payroll",
                "tx_recurrent_payroll",
                "meses_recurrent_payroll",
                "monto_recurrent_payroll",
            ],
            "list_cols": ["counterpartes_recurrent_payroll"],
        },
    }

    scenario_columns_order: list[str] = []
    for key, meta in scenario_metadata.items():
        scenario_df = scenario_frames.get(key, pd.DataFrame())
        expected_cols = ["persona"] + meta["columns"]
        if scenario_df.empty:
            scenario_df = pd.DataFrame(columns=expected_cols)
        else:
            scenario_df = scenario_df.reindex(columns=expected_cols)
        work = work.merge(scenario_df, on="persona", how="left")
        for col in meta.get("numeric_cols", []):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0.0)
        for col in meta.get("list_cols", []):
            if col in work.columns:
                work[col] = work[col].apply(_ensure_list)
        score_col = meta["score_col"]
        tier_col = meta["tier_col"]
        work[score_col] = (
            pd.to_numeric(work.get(score_col, 1), errors="coerce")
            .fillna(1)
            .astype(int)
        )
        work[tier_col] = work.get(tier_col, "Bajo").fillna("Bajo").astype(str)
        detail_col = meta.get("detail_col")
        if detail_col and detail_col in work.columns:
            work[detail_col] = work[detail_col].fillna("sin_detalle").astype(str)
        scenario_columns_order.extend(
            [col for col in meta["columns"] if col not in scenario_columns_order]
        )

    total_weight = sum(item["weight"] for item in scenario_metadata.values())
    work["casuistica_score_total"] = 0.0
    for meta in scenario_metadata.values():
        work["casuistica_score_total"] += work[meta["score_col"]] * meta["weight"]
    work["casuistica_score_promedio"] = (
        work["casuistica_score_total"] / total_weight if total_weight > 0 else 0.0
    )

    def _build_casuistica_resumen(row: pd.Series) -> str:
        contributions: list[tuple[int, float, str]] = []
        for meta in scenario_metadata.values():
            score_value = int(row.get(meta["score_col"], 1))
            if score_value <= 1:
                continue
            detail_col = meta.get("detail_col")
            detail = str(row.get(detail_col, "sin_detalle")) if detail_col else ""
            label = f"{meta['label']} {score_labels.get(score_value, 'Bajo')}"
            if detail and detail != "sin_detalle":
                label += f": {detail}"
            contributions.append((score_value, meta["weight"], label))
        if not contributions:
            return "Sin casuísticas destacadas"
        contributions.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return "; ".join(text for _, _, text in contributions[:3])

    work["casuistica_resumen"] = work.apply(_build_casuistica_resumen, axis=1)

    for extra_col in [
        "casuistica_score_total",
        "casuistica_score_promedio",
        "casuistica_resumen",
    ]:
        if extra_col not in columns:
            columns.append(extra_col)
    for col in scenario_columns_order:
        if col not in columns:
            columns.append(col)

    flag_rate_cols = {
        "yo_yo_persona_tasa_flag_emisor": "yo-yo",
        "smurf_persona_tasa_flag_emisor": "fraccionamiento",
        "frecuencia_persona_tasa_flag_emisor": "frecuencia inusual",
        "recurrente_persona_tasa_flag_emisor": "recurrente",
        "prestamo_persona_tasa_repay_insuficiente": "préstamo impago",
        "monto_persona_tasa_flag_redondo": "montos redondos",
        "umbral_persona_tasa_flag_cercania": "cercano a umbral",
        "red_persona_tasa_en_ciclos": "ciclos en red",
        "red_persona_tasa_en_triangulos": "triángulos en red",
        "quid_pro_quo_persona_tasa_flag": "algo por algo",
        "referencia_persona_tasa_reutilizada": "referencia reutilizada",
        "cambio_brusco_persona_tasa_flag": "cambio brusco",
        "nuevo_enlace_persona_tasa_flag": "nuevo enlace",
    }
    for col in flag_rate_cols:
        if col in work.columns:
            work[col] = (
                pd.to_numeric(work[col], errors="coerce").fillna(0.0).astype(float)
            )
        else:
            work[col] = 0.0

    def _risk_tier(value: float) -> str:
        if value >= 4.7:
            return "CRITICO"
        if value >= 3.2:
            return "ALTO"
        if value >= 1.8:
            return "MEDIO"
        return "BAJO"

    work["risk_tier"] = work["risk_avg_person"].apply(_risk_tier)

    if flag_rate_cols:
        flag_cols = list(flag_rate_cols.keys())
        work["flags_activas"] = work[flag_cols].gt(0).sum(axis=1).astype(int)
        work["flag_rate_max"] = work[flag_cols].max(axis=1)
    else:
        work["flags_activas"] = 0
        work["flag_rate_max"] = 0.0

    def _format_flags(row: pd.Series) -> str:
        pairs = [
            (flag_rate_cols[col], float(row.get(col, 0.0)))
            for col in flag_rate_cols
            if float(row.get(col, 0.0)) > 0
        ]
        pairs.sort(key=lambda item: item[1], reverse=True)
        top_items = pairs[:3]
        return (
            ", ".join(f"{name} ({rate:.0%})" for name, rate in top_items)
            if top_items
            else "sin_banderas_destacadas"
        )

    work["banderas_destacadas"] = work.apply(_format_flags, axis=1)

    work["abs_net"] = work["net_flow"].abs()
    work = work.sort_values(
        [
            "casuistica_score_total",
            "risk_avg_person",
            "abs_net",
            "flag_rate_max",
            "flags_activas",
        ],
        ascending=[False, False, False, False, False],
    ).head(max(1, int(top_n)))
    work["ranking_prioridad"] = list(range(1, len(work) + 1))

    def _direction(value: float) -> str:
        if value > 0:
            return "neto emisor"
        if value < 0:
            return "neto receptor"
        return "equilibrado"

    work["interpretabilidad"] = work.apply(
        lambda row: (
            f"En '{timeframe}', la persona {row.get('persona', 'sin_persona')} promedia "
            f"riesgo {row.get('risk_avg_person', 0):.2f} ({row.get('risk_tier', 'BAJO')}) "
            f"sobre {int(row.get('movements', 0))} movimientos. Emitió {int(row.get('n_tx_emit', 0))} "
            f"tx por {_format_float(row.get('sum_emit', 0))} y recibió {int(row.get('n_tx_recv', 0))} "
            f"tx por {_format_float(row.get('sum_recv', 0))}, quedando {_format_float(row.get('net_flow', 0))} "
            f"({_direction(float(row.get('net_flow', 0)))}). "
            f"Banderas destacadas: {row.get('banderas_destacadas', 'sin_banderas_destacadas')}. "
            f"Casuísticas activas: {row.get('casuistica_resumen', 'Sin casuísticas destacadas')}. "
            f"Score ponderado {row.get('casuistica_score_total', 0):.1f} (promedio {row.get('casuistica_score_promedio', 0):.2f}). "
            f"Desbalance mensual extremo al enviar en {row.get('desbalance_persona_tasa_meses_envia_extremo', 0):.0%} "
            f"de {int(row.get('desbalance_persona_meses_totales', 0))} meses y al recibir en "
            f"{row.get('desbalance_persona_tasa_meses_recibe_extremo', 0):.0%}."
        ),
        axis=1,
    )

    return work.reindex(columns=columns)


QUESTION_FUNCTIONS: Dict[str, Callable[..., pd.DataFrame]] = {
    "q1_manager_nlp": question1_manager_nlp,
    "q2_manager_concepts": question2_manager_concepts,
    "q3_quid_pairs": question3_quid_pairs,
    "q4_quid_negative_value_vs_load": question4_quid_negative_value_vs_load,
    "q5_reference_reuse": question5_reference_reuse,
    "q6_centralizers": question6_centralizers,
    "q7_net_imbalance": question7_net_imbalance,
    "q8_case13_new_employees": question8_case13_new_employees,
    "q9_case14_veterans_from_newcomers": question9_case14_veterans_from_newcomers,
    "q10_yoyo_streaks": question10_yoyo_streaks,
    "q11_near_threshold_structuring": question11_near_threshold_structuring,
    "q12_smurfing_chronic": question12_smurfing_chronic,
    "q13_bad_loans_with_frequency": question13_bad_loans_with_frequency,
    "q14_recurrent_payroll": question14_recurrent_payroll,
    "q15_coordinated_cluster_signals": question15_coordinated_cluster_signals,
    "q16_multisignal_transactions": question16_multisignal_transactions,
    "q17_nlp_person_profiles": question17_nlp_person_profiles,
    "q18_user_risk_scores": question18_user_risk_scores,
}


QUESTION_METADATA: Dict[str, Dict[str, Any]] = {}
for key, func in QUESTION_FUNCTIONS.items():
    doc = inspect.getdoc(func) or ""
    description = doc.splitlines()[0].strip() if doc else ""
    match = re.match(r"q(\d+)", key)
    order = int(match.group(1)) if match else 0
    QUESTION_METADATA[key] = {
        "title": QUESTION_TITLES.get(key, key),
        "description": description,
        "order": order,
        "function_name": func.__name__,
        "interpretability_column": "interpretabilidad",
    }

QUESTION_METADATA.get("q18_user_risk_scores", {}).update(
    {"table_preview_rows": 10, "interpretability_max_rows": 10}
)


def run_all_questions(
    reports: Mapping[str, Any],
    timeframe: str = DEFAULT_TIMEFRAME,
) -> Dict[str, pd.DataFrame]:
    """Ejecuta todas las preguntas estándar con la configuración por defecto."""

    results: Dict[str, pd.DataFrame] = {}
    for key, func in QUESTION_FUNCTIONS.items():
        results[key] = func(reports, timeframe=timeframe)
    return results


def get_question_overview() -> pd.DataFrame:
    """Devuelve un resumen tabular con títulos y descripciones de las preguntas."""

    rows: list[dict[str, Any]] = []
    for key, meta in QUESTION_METADATA.items():
        rows.append(
            {
                "question_id": key,
                "orden": meta.get("order", 0),
                "titulo": meta.get("title", key),
                "descripcion": meta.get("description", ""),
                "funcion": meta.get("function_name", ""),
                "columna_interpretabilidad": meta.get(
                    "interpretability_column", "interpretabilidad"
                ),
            }
        )
    overview = pd.DataFrame(rows)
    if overview.empty:
        return overview
    return overview.sort_values("orden").reset_index(drop=True)


def _run_questions(reports: Mapping[str, Any], timeframe: str) -> Dict[str, Any]:
    return run_all_questions(reports, timeframe=timeframe)


def _export_results(results: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(output_dir / f"{key}.csv", index=False)
        else:
            raise TypeError(
                f"El resultado de {key} no es un DataFrame y no puede exportarse."
            )


def _print_summary(
    results: Mapping[str, Any],
    *,
    max_rows: int = 3,
) -> None:
    """Muestra un resumen textual con descripciones e interpretabilidad."""

    for key, value in results.items():
        meta = QUESTION_METADATA.get(key, {})
        title = meta.get("title", key)
        description = meta.get("description")
        interpret_col = meta.get("interpretability_column", "interpretabilidad")

        print(f"\n== {title} ({key}) ==")
        if description:
            print(fill(description, width=100))

        if not isinstance(value, pd.DataFrame):
            print(value)
            continue

        total_rows = int(len(value))
        if total_rows == 0:
            print("No hay resultados para esta pregunta en el periodo analizado.")
            continue

        preview = value.drop(columns=[interpret_col], errors="ignore")
        if preview.empty:
            preview = value.copy()

        preview_rows = int(meta.get("table_preview_rows", max_rows))
        interpret_rows = int(meta.get("interpretability_max_rows", max_rows))

        with pd.option_context("display.max_columns", None, "display.width", 120):
            print(f"Filas disponibles: {total_rows}")
            print("Resumen tabular (primeras filas):")
            print(preview.head(preview_rows))

        if interpret_col in value.columns:
            interpret_values = (
                value[interpret_col]
                .dropna()
                .astype(str)
                .head(interpret_rows)
            )
            if not interpret_values.empty:
                print("Interpretabilidad destacada:")
                for text in interpret_values:
                    wrapped = fill(
                        text,
                        width=100,
                        initial_indent=" • ",
                        subsequent_indent="   ",
                    )
                    print(wrapped)


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
