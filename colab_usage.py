"""Utilities for running the COI/Fraud package directly from Google Colab.

The functions in this module provide a careful, end-to-end workflow that goes
from cloning the repository in the Colab filesystem to exporting every
``casuistica`` report (for all available timeframes) as CSV files.

Typical usage inside a Colab notebook::

    from colab_usage import (
        ensure_colab_environment,
        clone_or_update_repo,
        install_minimum_dependencies,
        append_repo_to_sys_path,
        load_transactions_dataframe,
        run_full_pipeline,
        export_casuistica_csv,
    )

    ensure_colab_environment()
    repo_path = clone_or_update_repo("https://github.com/tu-org/coi.git")
    install_minimum_dependencies()
    append_repo_to_sys_path(repo_path)

    # Replace this with your own CSV path in /content or in Drive
    df = load_transactions_dataframe("/content/mis_transacciones.csv")
    reports = run_full_pipeline(df)

    export_dir = "/content/resultados_casuistica"
    export_casuistica_csv(reports, export_dir)

The functions are intentionally explicit and check every precondition to reduce
unexpected failures while running inside Colab.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping

import pandas as pd

COLAB_ROOT = Path("/content")
CASUISTICA_PREFIX = "casuistica_"


class ColabUsageError(RuntimeError):
    """Custom error for Colab usage misconfigurations."""


def ensure_colab_environment() -> None:
    """Raise an error if the current interpreter is not running inside Colab."""

    if "google.colab" not in sys.modules:
        raise ColabUsageError(
            "Este módulo está pensado exclusivamente para Google Colab. "
            "Asegúrate de importar `colab_usage` dentro de un notebook de Colab."
        )


def _run_command(command: Iterable[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Execute a shell command and return the completed process."""

    process = subprocess.run(
        list(command),
        check=False,
        capture_output=True,
        text=True,
    )
    if check and process.returncode != 0:
        raise ColabUsageError(
            "El comando falló.\n"
            f"Comando: {' '.join(command)}\n"
            f"Código de salida: {process.returncode}\n"
            f"STDOUT: {process.stdout}\n"
            f"STDERR: {process.stderr}"
        )
    return process


def clone_or_update_repo(repo_url: str, target_dir: str | os.PathLike[str] | None = None) -> Path:
    """Clone the repository under /content or update it if it already exists."""

    ensure_colab_environment()

    if target_dir is None:
        target_path = COLAB_ROOT / "coi"
    else:
        target_path = Path(target_dir)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        # Fetch updates if the directory already exists.
        git_dir = target_path / ".git"
        if not git_dir.exists():
            raise ColabUsageError(
                f"El directorio destino {target_path} ya existe pero no contiene un repositorio Git."
            )
        _run_command(["git", "-C", str(target_path), "fetch", "--all"])
        _run_command(["git", "-C", str(target_path), "reset", "--hard", "origin/main"], check=False)
    else:
        _run_command(["git", "clone", repo_url, str(target_path)])
    return target_path


def install_minimum_dependencies(packages: Iterable[str] | None = None) -> None:
    """Install the minimal runtime dependencies required by the pipeline."""

    ensure_colab_environment()

    if packages is None:
        packages = ("pandas", "numpy", "seaborn", "scikit-learn", "scipy")
    _run_command([sys.executable, "-m", "pip", "install", "-q", *packages])


def append_repo_to_sys_path(repo_path: str | os.PathLike[str]) -> None:
    """Append the repository root (or its package folder) to ``sys.path``."""

    ensure_colab_environment()

    path = Path(repo_path)
    if not path.exists():
        raise ColabUsageError(f"La ruta {path} no existe en el entorno de Colab.")
    if str(path) not in sys.path:
        sys.path.append(str(path))


def load_transactions_dataframe(csv_path: str | os.PathLike[str]) -> pd.DataFrame:
    """Load a CSV file with the minimal transaction fields required by the pipeline."""

    path = Path(csv_path)
    if not path.exists():
        raise ColabUsageError(
            f"No se encontró el archivo CSV en {path}. Verifica que esté disponible en Colab."
        )
    df = pd.read_csv(path)
    required_columns = {
        "user_id",
        "receptor-user_id",
        "load_date",
        "movement_amount",
        "transaction_desc",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ColabUsageError(
            "El CSV carece de columnas obligatorias: " + ", ".join(sorted(missing))
        )
    return df


def run_full_pipeline(df: pd.DataFrame, *, language: str = "es") -> MutableMapping[str, Mapping[str, pd.DataFrame]]:
    """Execute ``coi_fraud.run_pipeline`` with the provided DataFrame."""

    ensure_colab_environment()

    try:
        from coi_fraud import run_pipeline
    except ImportError as exc:  # pragma: no cover - defensive branch for Colab setup
        raise ColabUsageError(
            "No se pudo importar `coi_fraud`. Asegúrate de haber añadido el repositorio "
            "al sys.path mediante `append_repo_to_sys_path` y de haber instalado las dependencias."
        ) from exc

    reports = run_pipeline(df, language=language)
    if not isinstance(reports, MutableMapping):
        raise ColabUsageError("El pipeline no devolvió un diccionario de reportes como se esperaba.")
    return reports


def export_casuistica_csv(
    reports: Mapping[str, Mapping[str, pd.DataFrame]],
    output_dir: str | os.PathLike[str],
    *,
    include_empty: bool = True,
) -> Dict[str, Dict[str, Path]]:
    """Export every casuística (for all timeframes) into CSV files.

    Parameters
    ----------
    reports:
        Diccionario devuelto por ``run_pipeline``. Se espera que cada entrada
        sea otro diccionario con los dataframes.
    output_dir:
        Directorio base donde se crearán los CSV.
    include_empty:
        Si es ``True``, también se generarán archivos CSV vacíos para casuísticas
        sin filas, manteniendo los encabezados.

    Returns
    -------
    Dict[str, Dict[str, Path]]
        Un diccionario con la ruta de cada CSV generado, organizado por
        casuística y timeframe.
    """

    ensure_colab_environment()

    base_path = Path(output_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    export_index: Dict[str, Dict[str, Path]] = {}
    for category, timeframe_map in reports.items():
        if not category.startswith(CASUISTICA_PREFIX):
            continue
        if not isinstance(timeframe_map, Mapping):
            continue
        export_index[category] = {}
        for timeframe, table in timeframe_map.items():
            if not isinstance(table, pd.DataFrame):
                continue
            output_path = base_path / f"{category}_{timeframe}.csv"
            if table.empty and not include_empty:
                continue
            table.to_csv(output_path, index=False)
            export_index[category][timeframe] = output_path
    return export_index


def export_index_to_json(index: Mapping[str, Mapping[str, Path]], destination: str | os.PathLike[str]) -> Path:
    """Persist the export index dictionary to disk for quick inspection."""

    ensure_colab_environment()

    dest_path = Path(destination)
    with dest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                category: {timeframe: str(path) for timeframe, path in timeframes.items()}
                for category, timeframes in index.items()
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    return dest_path


__all__ = [
    "CASUISTICA_PREFIX",
    "COLAB_ROOT",
    "ColabUsageError",
    "append_repo_to_sys_path",
    "clone_or_update_repo",
    "ensure_colab_environment",
    "export_casuistica_csv",
    "export_index_to_json",
    "install_minimum_dependencies",
    "load_transactions_dataframe",
    "run_full_pipeline",
]
