"""Utilidades específicas para ejecutar el paquete en Google Colab.

Este módulo está pensado para usarse directamente desde un notebook de
Google Colab. Incluye funciones para clonar el repositorio desde GitHub,
instalar dependencias mínimas, añadir el paquete al ``sys.path`` y
automatizar la generación de archivos CSV para cada casuística calculada
por :func:`coi_fraud.run_pipeline`.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import fill
from typing import Dict, Iterable, Mapping

import pandas as pd

try:  # pragma: no cover - solo válido en Colab con IPython
    from IPython import get_ipython
except ImportError:  # pragma: no cover - entorno estándar
    get_ipython = None  # type: ignore


DEFAULT_REPO_URL = "https://github.com/tu-org/coi.git"
DEFAULT_BRANCH = "main"
DEFAULT_TARGET_DIR = Path("/content/coi")
DEFAULT_PACKAGES = ("pandas", "numpy", "seaborn", "scikit-learn", "scipy")
CASUISTICA_PREFIX = "casuistica_"


def _is_colab() -> bool:
    """Detecta si el código se está ejecutando dentro de Google Colab."""

    if "COLAB_GPU" in os.environ:
        return True
    if get_ipython is None:
        return False
    shell = get_ipython()
    if not shell:
        return False
    return "google.colab" in str(type(shell))


def clone_repo(
    repo_url: str = DEFAULT_REPO_URL,
    target_dir: Path | str = DEFAULT_TARGET_DIR,
    branch: str = DEFAULT_BRANCH,
    *,
    force_refresh: bool = False,
) -> Path:
    """Clona el repositorio en ``target_dir`` si no existe.

    Parameters
    ----------
    repo_url:
        URL del repositorio GitHub.
    target_dir:
        Carpeta de destino dentro de ``/content``.
    branch:
        Rama a clonar.
    force_refresh:
        Si es ``True`` y la carpeta ya existe, se elimina para forzar una
        descarga limpia.
    """

    target_path = Path(target_dir)
    if target_path.exists():
        if force_refresh:
            shutil.rmtree(target_path)
        else:
            return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            branch,
            repo_url,
            str(target_path),
        ]
    )
    return target_path


def ensure_packages(packages: Iterable[str] | None = None) -> Iterable[str]:
    """Instala los paquetes listados si no están disponibles."""

    packages = tuple(packages or DEFAULT_PACKAGES)
    missing = [pkg for pkg in packages if importlib.util.find_spec(pkg) is None]
    if not missing:
        return ()
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *missing]
    )
    return missing


def add_repo_to_path(target_dir: Path | str = DEFAULT_TARGET_DIR) -> Path:
    """Asegura que ``target_dir`` esté en ``sys.path`` para importar ``coi_fraud``."""

    path = Path(target_dir)
    candidate = path if path.is_dir() else path.parent
    if str(candidate) not in sys.path:
        sys.path.append(str(candidate))
    return candidate


def setup_environment(
    repo_url: str = DEFAULT_REPO_URL,
    target_dir: Path | str = DEFAULT_TARGET_DIR,
    branch: str = DEFAULT_BRANCH,
    packages: Iterable[str] | None = None,
    force_refresh: bool = False,
) -> Path:
    """Pipeline completo de preparación del entorno para Colab.

    Devuelve la ruta al directorio donde quedó el repositorio.
    """

    if not _is_colab():
        raise RuntimeError(
            "Este helper está diseñado para Google Colab. Ejecuta el cuaderno "
            "en Colab antes de usar setup_environment()."
        )
    repo_path = clone_repo(repo_url=repo_url, target_dir=target_dir, branch=branch, force_refresh=force_refresh)
    ensure_packages(packages)
    add_repo_to_path(repo_path)
    return repo_path


def run_pipeline_from_csv(
    csv_path: str | os.PathLike,
    *,
    repo_dir: Path | str = DEFAULT_TARGET_DIR,
) -> Mapping[str, Dict[str, pd.DataFrame]]:
    """Carga un CSV, ejecuta ``run_pipeline`` y devuelve los reportes."""

    add_repo_to_path(repo_dir)
    from coi_fraud import run_pipeline  # import local tras asegurar sys.path

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("El archivo CSV está vacío; agrega transacciones antes de continuar.")
    reports = run_pipeline(df)
    if not isinstance(reports, Mapping):
        raise TypeError("run_pipeline debería devolver un mapeo de reportes.")
    return reports


def export_casuistica_to_csv(
    reports: Mapping[str, Mapping[str, pd.DataFrame]],
    output_dir: Path | str = Path("/content/coi_casuisticas"),
    *,
    include_empty: bool = False,
) -> Dict[str, Dict[str, Path]]:
    """Genera un CSV por casuística y periodo temporal.

    Parameters
    ----------
    reports:
        Diccionario devuelto por :func:`run_pipeline_from_csv`.
    output_dir:
        Carpeta base donde se crearán subdirectorios por casuística.
    include_empty:
        Si es ``True``, exporta también dataframes vacíos para dejar
        constancia de que no hubo hallazgos.

    Returns
    -------
    dict
        Estructura ``{casuistica: {timeframe: Path}}`` con las rutas de los
        archivos generados.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated: Dict[str, Dict[str, Path]] = {}
    for section_name, timeframes in reports.items():
        if not section_name.startswith(CASUISTICA_PREFIX):
            continue
        section_dir = output_path / section_name
        section_dir.mkdir(exist_ok=True)
        generated[section_name] = {}
        for timeframe, table in timeframes.items():
            if not isinstance(table, pd.DataFrame):
                continue
            if table.empty and not include_empty:
                continue
            filename = f"{section_name}_{timeframe}.csv"
            file_path = section_dir / filename
            table.to_csv(file_path, index=False)
            generated[section_name][timeframe] = file_path
    return generated


def run_full_colab_flow(
    csv_input_path: str | os.PathLike,
    *,
    repo_url: str = DEFAULT_REPO_URL,
    branch: str = DEFAULT_BRANCH,
    target_dir: Path | str = DEFAULT_TARGET_DIR,
    output_dir: Path | str = Path("/content/coi_casuisticas"),
    include_empty: bool = False,
    force_refresh: bool = False,
) -> Dict[str, Dict[str, Path]]:
    """Ejecuta todo el flujo: clonar, instalar, correr pipeline y exportar CSV."""

    repo_path = setup_environment(
        repo_url=repo_url,
        branch=branch,
        target_dir=target_dir,
        force_refresh=force_refresh,
    )
    reports = run_pipeline_from_csv(csv_input_path, repo_dir=repo_path)
    return export_casuistica_to_csv(
        reports,
        output_dir=output_dir,
        include_empty=include_empty,
    )


def question_overview(repo_dir: Path | str = DEFAULT_TARGET_DIR) -> pd.DataFrame:
    """Obtiene títulos, descripciones y metadatos de todas las preguntas Q1–Q18."""

    add_repo_to_path(repo_dir)
    from experiment_questions import get_question_overview

    overview = get_question_overview()
    if overview.empty:
        return overview
    columns = [
        "orden",
        "question_id",
        "titulo",
        "descripcion",
        "funcion",
        "columna_interpretabilidad",
    ]
    return overview.reindex(columns=[col for col in columns if col in overview.columns])


def summarize_question_interpretability(
    reports: Mapping[str, Dict[str, pd.DataFrame]],
    timeframe: str = DEFAULT_TIMEFRAME,
    *,
    max_rows: int = 3,
) -> pd.DataFrame:
    """Resume ejemplos de interpretabilidad para cada pregunta estándar.

    Parameters
    ----------
    reports:
        Diccionario devuelto por :func:`run_pipeline` con todas las secciones
        necesarias para las preguntas.
    timeframe:
        Ventana temporal a consultar (por defecto ``"todo_el_tiempo"``).
    max_rows:
        Número máximo de ejemplos de interpretabilidad a mostrar por pregunta.

    Returns
    -------
    pandas.DataFrame
        Tabla ordenada por ``orden`` con columnas de contexto e interpretabilidad
        condensada en formato multilínea.
    """

    from experiment_questions import QUESTION_METADATA, run_all_questions

    results = run_all_questions(reports, timeframe=timeframe)
    rows: list[dict[str, object]] = []
    for key, df in results.items():
        meta = QUESTION_METADATA.get(key, {})
        interpret_col = meta.get("interpretability_column", "interpretabilidad")
        total_rows = 0
        examples = "Sin resultados disponibles."
        columnas_clave = ""

        if isinstance(df, pd.DataFrame):
            total_rows = int(len(df))
            if not df.empty:
                display_cols = [col for col in df.columns if col != interpret_col][:6]
                if display_cols:
                    columnas_clave = ", ".join(display_cols)
                if interpret_col in df.columns:
                    sample = (
                        df[interpret_col]
                        .dropna()
                        .astype(str)
                        .head(max_rows)
                    )
                    if not sample.empty:
                        formatted = [
                            fill(
                                text,
                                width=100,
                                initial_indent="• ",
                                subsequent_indent="  ",
                            )
                            for text in sample
                        ]
                        examples = "\n".join(formatted)
                    else:
                        examples = "Interpretabilidad sin contenido en las primeras filas."
                else:
                    examples = "La columna de interpretabilidad no está disponible."
            else:
                examples = "Sin filas para este periodo." if interpret_col in df.columns else examples

        rows.append(
            {
                "orden": meta.get("order", 0),
                "question_id": key,
                "titulo": meta.get("title", key),
                "descripcion": meta.get("description", ""),
                "filas": total_rows,
                "columna_interpretabilidad": interpret_col,
                "columnas_clave": columnas_clave,
                "interpretabilidad_ejemplos": examples,
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    ordered_cols = [
        "orden",
        "question_id",
        "titulo",
        "descripcion",
        "filas",
        "columnas_clave",
        "columna_interpretabilidad",
        "interpretabilidad_ejemplos",
    ]
    summary = summary.sort_values("orden").reset_index(drop=True)
    return summary.reindex(columns=ordered_cols)


__all__ = [
    "clone_repo",
    "ensure_packages",
    "add_repo_to_path",
    "setup_environment",
    "run_pipeline_from_csv",
    "export_casuistica_to_csv",
    "run_full_colab_flow",
    "question_overview",
    "summarize_question_interpretability",
]
