"""Pruebas para la pregunta 1 de managers con conceptos NLP."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_questions import DEFAULT_TIMEFRAME, question1_manager_nlp
from coi_fraud.schemas import COL_AMOUNT, COL_DESCRIPTION, COL_RECEIVER_ID, COL_SENDER_ID


def test_question1_manager_nlp_infers_relations_from_manager_columns() -> None:
    """Debe detectar pagos sospechosos aunque falte la columna ``relacion``."""

    df = pd.DataFrame(
        [
            {
                "month_id": "2024-03",
                COL_SENDER_ID: "MGR-007",
                COL_RECEIVER_ID: "EMP-900",
                COL_AMOUNT: 1_250.0,
                COL_DESCRIPTION: "Pago especial de soborno a subordinado",
                "manager_1_user_id": "MGR-007",
            }
        ]
    )
    reports = {"transaccion": {DEFAULT_TIMEFRAME: df}}

    result = question1_manager_nlp(reports, timeframe=DEFAULT_TIMEFRAME)

    assert not result.empty, "Se esperaba al menos un resultado con la relación inferida"
    row = result.iloc[0]
    assert row["manager_user_id"] == "MGR-007"
    assert row["subordinado_user_id"] == "EMP-900"
    assert "SOBORNO" in row["nlp_concepto_sospechoso"], row["nlp_concepto_sospechoso"]


def test_question1_manager_nlp_includes_both_directions_by_default() -> None:
    """El flujo por defecto debe conservar envíos en ambos sentidos sospechosos."""

    df = pd.DataFrame(
        [
            {
                "month_id": "2024-01",
                COL_SENDER_ID: "M1",
                COL_RECEIVER_ID: "E1",
                "relacion": "manager_del_receptor",
                COL_AMOUNT: 1_000.0,
                COL_DESCRIPTION: "Pago etiquetado como soborno",
                "nlp_concepto_sospechoso": "SOBORNO",
            },
            {
                "month_id": "2024-01",
                COL_SENDER_ID: "E2",
                COL_RECEIVER_ID: "M2",
                "relacion": "manager_del_emisor",
                COL_AMOUNT: 2_000.0,
                COL_DESCRIPTION: "Pago etiquetado como soborno",
                "nlp_concepto_sospechoso": "SOBORNO",
            },
        ]
    )
    reports = {"transaccion": {DEFAULT_TIMEFRAME: df}}

    result = question1_manager_nlp(reports, timeframe=DEFAULT_TIMEFRAME)

    assert set(result["flow_direction"]) == {
        "manager_a_subordinado",
        "subordinado_a_manager",
    }

    reverse = result.loc[result["flow_direction"] == "subordinado_a_manager"].iloc[0]
    assert reverse["manager_user_id"] == "M2"
    assert reverse["subordinado_user_id"] == "E2"
    assert "el subordinado e2 envió" in reverse["interpretabilidad"].lower()
