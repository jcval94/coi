"""Ejecución manual de question1_manager_nlp sin columna de relación."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_questions import DEFAULT_TIMEFRAME, question1_manager_nlp
from coi_fraud.schemas import COL_AMOUNT, COL_DESCRIPTION, COL_RECEIVER_ID, COL_SENDER_ID


def build_reports_without_relation() -> dict[str, dict[str, pd.DataFrame]]:
    """Construye reportes donde la relación manager-subordinado debe inferirse."""

    data = [
        {
            "month_id": "2024-03",
            COL_SENDER_ID: "MGR-500",
            COL_RECEIVER_ID: "EMP-301",
            COL_AMOUNT: 2_100.0,
            COL_DESCRIPTION: "Soborno especial acordado con subordinado",
            "manager_1_user_id": "MGR-500",
        },
        {
            "month_id": "2024-04",
            COL_SENDER_ID: "MGR-500",
            COL_RECEIVER_ID: "EMP-302",
            COL_AMOUNT: 950.0,
            COL_DESCRIPTION: "Pago de facilitación para mover contrato",
            "manager_1_user_id": "MGR-500",
        },
    ]
    df = pd.DataFrame(data)
    # La columna ``relacion`` no existe de forma deliberada para probar la inferencia.
    return {"transaccion": {DEFAULT_TIMEFRAME: df}}


def run(output_path: Path) -> pd.DataFrame:
    reports = build_reports_without_relation()
    result = question1_manager_nlp(reports, timeframe=DEFAULT_TIMEFRAME)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    output_path = Path("answers/test_question1_manager_nlp_missing_relation.csv")
    result = run(output_path)
    print(result)
    print(f"\nSe guardaron {len(result)} filas en '{output_path}'.")


if __name__ == "__main__":
    main()
