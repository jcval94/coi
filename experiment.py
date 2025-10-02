"""Script de experimento para ejecutar la canalización con datos sintéticos."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

try:
    import pandas as pd
except ModuleNotFoundError as exc:  # pragma: no cover - guard for entornos sin deps
    raise SystemExit(
        "Este script requiere pandas. Ejecuta 'pip install -r requirements.txt' "
        "para instalar las dependencias necesarias antes de correrlo."
    ) from exc

import numpy as np

from coi_fraud import generate_diverse_dataset, run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un dataset sintético diverso y ejecuta la canalización "
            "completa de COI/Fraud para producir un resumen rápido."
        )
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=6_000,
        help="Número de transacciones a generar (por defecto: 6000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla numérica para replicar el experimento (por defecto: 42).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Cantidad de personas con mayor riesgo a mostrar (por defecto: 5).",
    )
    return parser


def _ensure_column(frame: pd.DataFrame, target: str, candidates: list[str], fill_value: float) -> None:
    if target in frame.columns:
        return
    for candidate in candidates:
        if candidate in frame.columns:
            frame[target] = frame[candidate]
            return
    frame[target] = fill_value


INTERVAL_LABELS: Dict[str, str] = {
    "ultimo_mes": "último mes",
    "ultimos_3_meses": "últimos 3 meses",
    "todo_el_tiempo": "todo el tiempo",
}


def _select_columns(frame: pd.DataFrame, desired: list[str]) -> list[str]:
    return [column for column in desired if column in frame.columns]


def _summarize_persona(frame: pd.DataFrame | None, top_n: int) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()

    persona_frame = frame.copy()
    if persona_frame.empty:
        return persona_frame

    _ensure_column(
        persona_frame,
        "risk_avg_person",
        [
            "risk_avg_person",
            "risk_avg_em",
            "risk_max_emit",
            "avg_emit",
            "avg_recv",
        ],
        fill_value=0.0,
    )
    _ensure_column(
        persona_frame,
        "sum_emit",
        ["sum_emit", "tx_sum_emit", "tx_sum"],
        fill_value=0.0,
    )
    _ensure_column(
        persona_frame,
        "sum_recv",
        ["sum_recv", "tx_sum_recv"],
        fill_value=0.0,
    )

    if "movements" not in persona_frame.columns:
        if {"n_tx_emit", "n_tx_recv"}.issubset(persona_frame.columns):
            persona_frame["movements"] = (
                persona_frame["n_tx_emit"].fillna(0) + persona_frame["n_tx_recv"].fillna(0)
            )
        elif "n_tx" in persona_frame.columns:
            persona_frame["movements"] = persona_frame["n_tx"].fillna(0)
        else:
            persona_frame["movements"] = np.nan

    ordered = persona_frame.sort_values("risk_avg_person", ascending=False)
    columns = _select_columns(
        ordered,
        [
            "persona",
            "risk_avg_person",
            "sum_emit",
            "sum_recv",
            "movements",
            "interp_person",
        ],
    )
    return ordered.head(top_n)[columns]


def _summarize_pairs(frame: pd.DataFrame | None, top_n: int) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()

    pairs_frame = frame.copy()
    if pairs_frame.empty:
        return pairs_frame

    _ensure_column(
        pairs_frame,
        "risk_avg_pair",
        ["risk_avg_pair", "risk_max", "risk_avg"],
        fill_value=0.0,
    )
    _ensure_column(
        pairs_frame,
        "sum_emit",
        ["sum_emit", "tx_sum_emit", "tx_sum"],
        fill_value=0.0,
    )
    _ensure_column(
        pairs_frame,
        "sum_recv",
        ["sum_recv", "tx_sum_recv"],
        fill_value=0.0,
    )
    _ensure_column(
        pairs_frame,
        "movements",
        ["movements", "tx_count", "n_tx"],
        fill_value=0.0,
    )

    ordered = pairs_frame.sort_values("risk_avg_pair", ascending=False)
    columns = _select_columns(
        ordered,
        [
            "pair",
            "risk_avg_pair",
            "sum_emit",
            "sum_recv",
            "movements",
            "interp_pair",
        ],
    )
    return ordered.head(top_n)[columns]


def _summarize_transactions(frame: pd.DataFrame | None, top_n: int) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()

    tx_frame = frame.copy()
    if tx_frame.empty:
        return tx_frame

    _ensure_column(
        tx_frame,
        "risk_score",
        ["risk_score", "risk_score_norm", "risk"],
        fill_value=0.0,
    )

    ordered = tx_frame.sort_values("risk_score", ascending=False)
    columns = _select_columns(
        ordered,
        [
            "fecha_hora_ts",
            "user_id",
            "receptor-user_id",
            "movement_amount",
            "risk_score",
            "descripcion",
            "interp_tx",
        ],
    )
    return ordered.head(top_n)[columns]


def _summarize_clusters(frame: pd.DataFrame | None, top_n: int) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()

    clusters_frame = frame.copy()
    if clusters_frame.empty:
        return clusters_frame

    _ensure_column(
        clusters_frame,
        "riesgo_cluster_maximo",
        ["riesgo_cluster_maximo", "cluster_risk_max", "risk_max"],
        fill_value=0.0,
    )

    ordered = clusters_frame.sort_values("riesgo_cluster_maximo", ascending=False)
    columns = _select_columns(
        ordered,
        [
            "cluster_id",
            "cluster_personas_total",
            "cluster_tx_count",
            "cluster_tx_sum",
            "riesgo_cluster_maximo",
            "riesgo_cluster_promedio",
            "interp_cluster",
        ],
    )
    return ordered.head(top_n)[columns]


def _summarize_concepts(frame: pd.DataFrame | None, top_n: int) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()

    concept_frame = frame.copy()
    if concept_frame.empty:
        return concept_frame

    _ensure_column(
        concept_frame,
        "nlp_concepto_riesgo_promedio",
        ["nlp_concepto_riesgo_promedio", "risk_avg_concept"],
        fill_value=0.0,
    )

    ordered = concept_frame.sort_values(
        "nlp_concepto_riesgo_promedio", ascending=False
    )
    columns = _select_columns(
        ordered,
        [
            "nlp_concepto_sospechoso",
            "nlp_concepto_transacciones",
            "nlp_concepto_monto_total",
            "nlp_concepto_riesgo_promedio",
            "nlp_concepto_riesgo_p95",
        ],
    )
    return ordered.head(top_n)[columns]


def _summarize_person_concepts(
    frame: pd.DataFrame | None, top_n: int
) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()

    person_concept = frame.copy()
    if person_concept.empty:
        return person_concept

    _ensure_column(
        person_concept,
        "nlp_persona_concepto_riesgo_promedio",
        ["nlp_persona_concepto_riesgo_promedio", "risk_avg"],
        fill_value=0.0,
    )

    ordered = person_concept.sort_values(
        "nlp_persona_concepto_riesgo_promedio", ascending=False
    )
    columns = _select_columns(
        ordered,
        [
            "persona",
            "nlp_concepto_sospechoso",
            "nlp_persona_concepto_tx_total",
            "nlp_persona_concepto_monto_total",
            "nlp_persona_concepto_riesgo_promedio",
        ],
    )
    return ordered.head(top_n)[columns]


def _summarize_pair_concepts(frame: pd.DataFrame | None, top_n: int) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()

    pair_concept = frame.copy()
    if pair_concept.empty:
        return pair_concept

    _ensure_column(
        pair_concept,
        "nlp_par_concepto_riesgo_promedio",
        ["nlp_par_concepto_riesgo_promedio", "risk_avg"],
        fill_value=0.0,
    )

    ordered = pair_concept.sort_values(
        "nlp_par_concepto_riesgo_promedio", ascending=False
    )
    columns = _select_columns(
        ordered,
        [
            "pair",
            "nlp_concepto_sospechoso",
            "nlp_par_concepto_tx_total",
            "nlp_par_concepto_monto_total",
            "nlp_par_concepto_riesgo_promedio",
        ],
    )
    return ordered.head(top_n)[columns]


def _ensure_results_dir() -> Path:
    """Crea el directorio de resultados si no existe y lo devuelve."""

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def _export_sample(
    frame: pd.DataFrame,
    output_path: Path,
    sample_size: int,
    description: str,
) -> None:
    """Exporta un subconjunto del DataFrame a CSV si hay datos disponibles."""

    if frame.empty:
        print(f"⚠️  No hay datos para exportar en {description}.")
        return

    limited = frame.head(sample_size)
    limited.to_csv(output_path, index=False)
    print(f"💾 Muestra guardada en {output_path.resolve()}")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    results_dir = _ensure_results_dir()
    sample_size = max(1, min(args.top_n, 10))

    print("⚙️  Generando dataset sintético...")
    dataset = generate_diverse_dataset(n_records=args.rows, seed=args.seed)
    print(f"Dataset generado con {len(dataset):,} filas y {len(dataset.columns)} columnas.")

    print("🚀 Ejecutando pipeline de riesgo...")
    reports = run_pipeline(dataset)

    persona_section = reports.get("persona") or {}
    print("\n🏅 Personas con mayor riesgo promedio por intervalo:")
    persona_found = False
    for interval_key, interval_label in INTERVAL_LABELS.items():
        persona_top = _summarize_persona(persona_section.get(interval_key), args.top_n)
        if persona_top.empty:
            print(f" - No hay datos para {interval_label}.")
            continue

        persona_found = True
        print(f"\nIntervalo: {interval_label}")
        print(persona_top.to_string(index=False))
        _export_sample(
            persona_top,
            results_dir / f"personas_riesgo_{interval_key}.csv",
            sample_size,
            f"personas con mayor riesgo ({interval_label})",
        )

    if not persona_found:
        print("No se encontraron personas con transacciones en los periodos evaluados.")

    pair_section = reports.get("pair") or reports.get("par_personas") or {}
    print("\n🤝 Parejas emisores-receptores con mayor riesgo por intervalo:")
    pair_found = False
    for interval_key, interval_label in INTERVAL_LABELS.items():
        pair_top = _summarize_pairs(pair_section.get(interval_key), args.top_n)
        if pair_top.empty:
            print(f" - No hay datos para {interval_label}.")
            continue

        pair_found = True
        print(f"\nIntervalo: {interval_label}")
        print(pair_top.to_string(index=False))
        _export_sample(
            pair_top,
            results_dir / f"parejas_riesgo_{interval_key}.csv",
            sample_size,
            f"parejas con mayor riesgo ({interval_label})",
        )

    if not pair_found:
        print("No se encontraron pares con interacciones registradas en los periodos evaluados.")

    tx_section = reports.get("transaccion") or {}
    print("\n💸 Transacciones con mayor riesgo e interpretabilidad por intervalo:")
    tx_found = False
    for interval_key, interval_label in INTERVAL_LABELS.items():
        tx_top = _summarize_transactions(tx_section.get(interval_key), args.top_n)
        if tx_top.empty:
            print(f" - No hay transacciones con riesgo para {interval_label}.")
            continue

        tx_found = True
        print(f"\nIntervalo: {interval_label}")
        print(tx_top.to_string(index=False))
        _export_sample(
            tx_top,
            results_dir / f"transacciones_riesgo_{interval_key}.csv",
            sample_size,
            f"transacciones con mayor riesgo ({interval_label})",
        )

    if not tx_found:
        print("No se detectaron transacciones de riesgo para los periodos solicitados.")

    clusters_section = reports.get("clusters_personas") or {}
    print("\n🕸️  Clusters de personas con mayor riesgo e interpretabilidad por intervalo:")
    clusters_found = False
    for interval_key, interval_label in INTERVAL_LABELS.items():
        clusters_top = _summarize_clusters(clusters_section.get(interval_key), args.top_n)
        if clusters_top.empty:
            print(f" - No hay clusters relevantes para {interval_label}.")
            continue

        clusters_found = True
        print(f"\nIntervalo: {interval_label}")
        print(clusters_top.to_string(index=False))
        _export_sample(
            clusters_top,
            results_dir / f"clusters_riesgo_{interval_key}.csv",
            sample_size,
            f"clusters con mayor riesgo ({interval_label})",
        )

    if not clusters_found:
        print("No se encontraron clusters con actividad sospechosa en los periodos evaluados.")

    concepts_section = reports.get("concepto_descripcion") or {}
    print("\n🧠 Conceptos sospechosos por intervalo:")
    concepts_found = False
    for interval_key, interval_label in INTERVAL_LABELS.items():
        concepts_top = _summarize_concepts(concepts_section.get(interval_key), args.top_n)
        if concepts_top.empty:
            print(f" - No hay conceptos sospechosos para {interval_label}.")
            continue

        concepts_found = True
        print(f"\nIntervalo: {interval_label}")
        print(concepts_top.to_string(index=False))
        _export_sample(
            concepts_top,
            results_dir / f"conceptos_riesgo_{interval_key}.csv",
            sample_size,
            f"conceptos sospechosos ({interval_label})",
        )

    if not concepts_found:
        print("No se identificaron conceptos sospechosos en los periodos evaluados.")

    persona_concept_section = reports.get("persona_concepto") or {}
    print("\n🧾 Personas asociadas a conceptos sospechosos por intervalo:")
    persona_concept_found = False
    for interval_key, interval_label in INTERVAL_LABELS.items():
        persona_concept_top = _summarize_person_concepts(
            persona_concept_section.get(interval_key), args.top_n
        )
        if persona_concept_top.empty:
            print(f" - No hay asociaciones persona-concepto para {interval_label}.")
            continue

        persona_concept_found = True
        print(f"\nIntervalo: {interval_label}")
        print(persona_concept_top.to_string(index=False))
        _export_sample(
            persona_concept_top,
            results_dir / f"persona_conceptos_riesgo_{interval_key}.csv",
            sample_size,
            f"personas con conceptos sospechosos ({interval_label})",
        )

    if not persona_concept_found:
        print("No se encontraron personas vinculadas a conceptos sospechosos en los periodos evaluados.")

    pair_concept_section = reports.get("par_concepto") or {}
    print("\n🔗 Parejas asociadas a conceptos sospechosos por intervalo:")
    pair_concept_found = False
    for interval_key, interval_label in INTERVAL_LABELS.items():
        pair_concept_top = _summarize_pair_concepts(
            pair_concept_section.get(interval_key), args.top_n
        )
        if pair_concept_top.empty:
            print(f" - No hay asociaciones par-concepto para {interval_label}.")
            continue

        pair_concept_found = True
        print(f"\nIntervalo: {interval_label}")
        print(pair_concept_top.to_string(index=False))
        _export_sample(
            pair_concept_top,
            results_dir / f"pareja_conceptos_riesgo_{interval_key}.csv",
            sample_size,
            f"parejas con conceptos sospechosos ({interval_label})",
        )

    if not pair_concept_found:
        print("No se encontraron pares vinculados a conceptos sospechosos en los periodos evaluados.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
