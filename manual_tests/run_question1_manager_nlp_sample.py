"""Ejecución manual de question1_manager_nlp con un dataset pequeño de ejemplo."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Asegura que el directorio raíz del repositorio esté en sys.path para las importaciones
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_questions import DEFAULT_TIMEFRAME, question1_manager_nlp
from coi_fraud.schemas import (
    COL_AMOUNT,
    COL_DESCRIPTION,
    COL_RECEIVER_ID,
    COL_RELATION,
    COL_SENDER_ID,
)


def build_sample_reports() -> dict[str, dict[str, pd.DataFrame]]:
    """Construye un diccionario de reportes con transacciones sintéticas."""
    sample_data = [
        {
            "month_id": "2024-01",
            COL_SENDER_ID: "MGR001",
            COL_RECEIVER_ID: "EMP100",
            COL_RELATION: "manager_del_receptor",
            COL_AMOUNT: 1_500.0,
            COL_DESCRIPTION: "Pago especial por contrato",
            "nlp_concepto_sospechoso": "SOBORNO",
            "nlp_concepto_crudo": "Soborno en efectivo",
        },
        {
            "month_id": "2024-01",
            COL_SENDER_ID: "MGR001",
            COL_RECEIVER_ID: "EMP100",
            COL_RELATION: "manager_del_receptor",
            COL_AMOUNT: 800.0,
            COL_DESCRIPTION: "Trámite acelerado con apoyo",
            "nlp_concepto_sospechoso": "FACILITACIÓN",
            "nlp_concepto_crudo": "Pago de facilitación",
        },
        {
            "month_id": "2024-02",
            COL_SENDER_ID: "EMP200",
            COL_RECEIVER_ID: "MGR200",
            COL_RELATION: "manager_del_emisor",
            COL_AMOUNT: 1_200.0,
            COL_DESCRIPTION: "Pago coordinado reiterado",
            "nlp_concepto_sospechoso": "COORDINACION_REITERADA",
            "nlp_concepto_crudo": "Reiteramos lo pactado",
        },
        {
            "month_id": "2024-02",
            COL_SENDER_ID: "MGR500",
            COL_RECEIVER_ID: "EMP300",
            COL_RELATION: "relacion_manager_desconocida",
            COL_AMOUNT: 2_000.0,
            COL_DESCRIPTION: "Convivio exclusivo con equipo",
            "nlp_concepto_sospechoso": "AGASAJOS_SOCIALES",
            "nlp_concepto_crudo": "Evento social privado",
        },
    ]
    df = pd.DataFrame(sample_data)
    return {"transaccion": {DEFAULT_TIMEFRAME: df}}


def run(output_path: Path) -> pd.DataFrame:
    """Ejecuta la función con los datos de ejemplo y guarda el resultado."""
    reports = build_sample_reports()
    result = question1_manager_nlp(reports, timeframe=DEFAULT_TIMEFRAME)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    output_path = Path("answers/test_question1_manager_nlp_sample.csv")
    result = run(output_path)
    print(result)
    print(f"\nSe guardaron {len(result)} filas en '{output_path}'.")


if __name__ == "__main__":
    main()
