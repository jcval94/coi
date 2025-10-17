"""Pruebas para la pregunta 1 de managers con conceptos NLP."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_questions import DEFAULT_TIMEFRAME, question1_manager_nlp
from coi_fraud.schemas import (
    COL_AMOUNT,
    COL_DESCRIPTION,
    COL_RECEIVER_ID,
    COL_RELATION,
    COL_SENDER_ID,
)


def test_question1_manager_nlp_infers_manager_from_receiver_id() -> None:
    """Identifica managers cuando el receptor aparece en columnas jerárquicas."""

    df = pd.DataFrame(
        [
            {
                "month_id": "2024-03",
                COL_SENDER_ID: "EMP-900",
                COL_RECEIVER_ID: "MGR-007",
                COL_AMOUNT: 1_250.0,
                COL_DESCRIPTION: "Pago especial de soborno a manager",
                "manager_1_user_id": "MGR-007",
                "companeros_de_equipo": "EMP-901|EMP-902",
            }
        ]
    )
    reports = {"transaccion": {DEFAULT_TIMEFRAME: df}}

    result = question1_manager_nlp(
        reports, timeframe=DEFAULT_TIMEFRAME, direction="subordinado_a_manager"
    )

    assert not result.empty, "Se esperaba al menos un resultado con la relación inferida"
    row = result.iloc[0]
    assert row["manager_user_id"] == "MGR-007"
    assert row["subordinado_user_id"] == "EMP-900"
    assert row["direction"] == "subordinado_a_manager"
    assert "SOBORNO" in row["nlp_concepto_sospechoso"], row["nlp_concepto_sospechoso"]


def test_question1_manager_nlp_handles_empty_relation_column() -> None:
    """Cuando ``relacion`` no menciona managers debe inferir la orientación."""

    df = pd.DataFrame(
        [
            {
                "month_id": "2024-04",
                COL_SENDER_ID: "MGR-900",
                COL_RECEIVER_ID: "EMP-123",
                COL_AMOUNT: 2_500.0,
                COL_DESCRIPTION: "Transferencia clasificada como soborno",
                "receptor-manager_1_user_id": "MGR-900",
                "receptor-companeros_de_equipo": "EMP-200,EMP-201",
                # La columna existe pero no aporta información útil sobre managers.
                COL_RELATION: "transferencia_interna",
                "nlp_concepto_sospechoso": "SOBORNO",
            }
        ]
    )
    reports = {"transaccion": {DEFAULT_TIMEFRAME: df}}

    result = question1_manager_nlp(reports, timeframe=DEFAULT_TIMEFRAME)

    assert not result.empty, "Debe detectar coincidencias aunque 'relacion' no indique manager"
    row = result.iloc[0]
    assert row["manager_user_id"] == "MGR-900"
    assert row["subordinado_user_id"] == "EMP-123"
    assert row["direction"] == "manager_a_subordinado"


def test_question1_manager_nlp_excludes_teammates_from_manager_match() -> None:
    """No debe clasificar como manager cuando solo existe relación de compañeros."""

    df = pd.DataFrame(
        [
            {
                "month_id": "2024-05",
                COL_SENDER_ID: "EMP-500",
                COL_RECEIVER_ID: "EMP-600",
                COL_AMOUNT: 900.0,
                COL_DESCRIPTION: "Transferencia rutinaria",
                "manager_1_user_id": "MGR-111",
                "companeros_de_equipo": "EMP-600;EMP-601",
                "nlp_concepto_sospechoso": "SOBORNO",
            }
        ]
    )
    reports = {"transaccion": {DEFAULT_TIMEFRAME: df}}

    result = question1_manager_nlp(
        reports, timeframe=DEFAULT_TIMEFRAME, direction="subordinado_a_manager"
    )

    assert result.empty, "No debe detectar relación manager-subordinado entre compañeros"


def test_question1_manager_nlp_fallbacks_to_detected_direction() -> None:
    """Cuando solo hay flujo subordinado→manager debe devolver resultados."""

    df = pd.DataFrame(
        [
            {
                "month_id": "2024-06",
                COL_SENDER_ID: "EMP-500",
                COL_RECEIVER_ID: "MGR-321",
                COL_AMOUNT: 4_200.0,
                COL_DESCRIPTION: "Pago directo de soborno documentado",
                COL_RELATION: "manager_del_emisor",
                "nlp_concepto_sospechoso": "SOBORNO",
            }
        ]
    )
    reports = {"transaccion": {DEFAULT_TIMEFRAME: df}}

    result = question1_manager_nlp(reports, timeframe=DEFAULT_TIMEFRAME)

    assert not result.empty, "Debe aprovechar la orientación detectada en 'relacion'"
    row = result.iloc[0]
    assert row["manager_user_id"] == "MGR-321"
    assert row["subordinado_user_id"] == "EMP-500"
    assert row["direction"] == "subordinado_a_manager"
