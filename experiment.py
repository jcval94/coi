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
        default=5_000,
        help="Número de transacciones a generar (por defecto: 5000).",
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


def _summarize_persona(reports: Dict[str, Dict[str, pd.DataFrame]], top_n: int) -> pd.DataFrame:
    persona_mes = reports["persona"]["ultimo_mes"].copy()
    if persona_mes.empty:
        return persona_mes

    columns = [
        "persona",
        "risk_avg_person",
        "sum_emit",
        "sum_recv",
        "movements",
    ]
    for col in columns:
        if col not in persona_mes.columns:
            raise KeyError(
                f"La columna esperada '{col}' no está disponible en el reporte de persona."
            )

    ordered = persona_mes.sort_values("risk_avg_person", ascending=False)
    return ordered.head(top_n)[columns]


def _summarize_pairs(reports: Dict[str, Dict[str, pd.DataFrame]], top_n: int) -> pd.DataFrame:
    pairs_all = reports["pair"]["todo_el_tiempo"].copy()
    if pairs_all.empty:
        return pairs_all

    columns = [
        "pair",
        "risk_avg_pair",
        "sum_emit",
        "sum_recv",
        "movements",
    ]
    for col in columns:
        if col not in pairs_all.columns:
            raise KeyError(
                f"La columna esperada '{col}' no está disponible en el reporte de parejas."
            )

    ordered = pairs_all.sort_values("risk_avg_pair", ascending=False)
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

    print("\n🏅 Personas con mayor riesgo promedio en el último mes:")
    persona_top = _summarize_persona(reports, args.top_n)
    if persona_top.empty:
        print("No se encontraron personas con transacciones en el periodo evaluado.")
    else:
        print(persona_top.to_string(index=False))
        _export_sample(
            persona_top,
            results_dir / "personas_riesgo_muestra.csv",
            sample_size,
            "personas con mayor riesgo",
        )

    print("\n🤝 Parejas emisores-receptores con mayor riesgo acumulado:")
    pair_top = _summarize_pairs(reports, args.top_n)
    if pair_top.empty:
        print("No se encontraron pares con interacciones registradas.")
    else:
        print(pair_top.to_string(index=False))
        _export_sample(
            pair_top,
            results_dir / "parejas_riesgo_muestra.csv",
            sample_size,
            "parejas con mayor riesgo",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
