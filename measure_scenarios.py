"""Script para medir tiempos y filas generadas por cada escenario de Q&A."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from coi_fraud import generate_diverse_dataset, run_pipeline
from experiment_questions import (
    DEFAULT_TIMEFRAME,
    QUESTION_FUNCTIONS,
    QUESTION_METADATA,
)

DEFAULT_TIMEFRAMES: Sequence[str] = ("ultimo_mes", "ultimos_3_meses", DEFAULT_TIMEFRAME)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un dataset sintético, ejecuta la canalización de riesgo "
            "y evalúa cada escenario/pregunta Q1–Q17 midiendo tiempos y filas."
        )
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=10_000,
        help="Número de transacciones sintéticas a generar (por defecto: 10000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para el generador sintético (por defecto: 42).",
    )
    parser.add_argument(
        "--timeframes",
        nargs="*",
        default=list(DEFAULT_TIMEFRAMES),
        choices=list(DEFAULT_TIMEFRAMES),
        help=(
            "Ventanas temporales a evaluar. Si no se especifica se usan las tres "
            "disponibles (último mes, últimos 3 meses y todo el tiempo)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/escenarios_metricas.csv"),
        help="Archivo CSV donde guardar el resumen de métricas (por defecto: results/escenarios_metricas.csv).",
    )
    return parser


def _ensure_timeframes(timeframes: Iterable[str]) -> list[str]:
    unique: list[str] = []
    for timeframe in timeframes:
        if timeframe not in unique:
            unique.append(timeframe)
    return unique or list(DEFAULT_TIMEFRAMES)


def measure_scenarios(rows: int, seed: int, timeframes: Iterable[str]) -> pd.DataFrame:
    dataset = generate_diverse_dataset(n_records=rows, seed=seed)

    pipeline_start = time.perf_counter()
    reports = run_pipeline(dataset)
    pipeline_duration = time.perf_counter() - pipeline_start

    metrics: list[dict[str, object]] = [
        {
            "escenario": "pipeline",
            "titulo": "Ejecución de pipeline",
            "ventana": "-",
            "duracion_segundos": pipeline_duration,
            "filas": len(dataset),
        }
    ]

    for timeframe in timeframes:
        for key, func in QUESTION_FUNCTIONS.items():
            title = QUESTION_METADATA.get(key, {}).get("title", key)
            start = time.perf_counter()
            result = func(reports, timeframe=timeframe)
            duration = time.perf_counter() - start
            if isinstance(result, pd.DataFrame):
                row_count = int(len(result))
            else:
                row_count = 0
            metrics.append(
                {
                    "escenario": key,
                    "titulo": title,
                    "ventana": timeframe,
                    "duracion_segundos": duration,
                    "filas": row_count,
                }
            )

    return pd.DataFrame(metrics)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    timeframes = _ensure_timeframes(args.timeframes)
    summary = measure_scenarios(rows=args.rows, seed=args.seed, timeframes=timeframes)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print("Resumen de métricas por escenario:")
        print(summary)
        print(f"\nCSV guardado en {output_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
