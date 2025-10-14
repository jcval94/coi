import io
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from coi_fraud.schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_SENDER_ID
from experiment_questions import DEFAULT_TIMEFRAME, question5_reference_reuse


def _build_reports(transactions: pd.DataFrame) -> dict:
    return {"transaccion": {DEFAULT_TIMEFRAME: transactions}}


def test_question5_reference_reuse_to_csv_roundtrip(tmp_path):
    transactions = pd.DataFrame(
        {
            COL_SENDER_ID: ["S1", "S2", "S3", "S1"],
            COL_RECEIVER_ID: ["R1", "R1", "R1", "R2"],
            "nlp_concepto_sospechoso": [
                "Pago extraordinario",
                "Pago extraordinario",
                "Pago extraordinario",
                "Pago único",
            ],
            "nlp_concepto_crudo": [
                "Pago extra",
                "Pago extra",
                "Pago-Extra",
                "Único",
            ],
            COL_AMOUNT: [1000.0, 2000.0, 500.0, 700.0],
            "month_id": ["2024-01", "2024-02", "2024-02", "2024-01"],
            "risk_score": [0.2, 0.4, 0.6, 0.9],
        }
    )

    result = question5_reference_reuse(
        _build_reports(transactions),
        DEFAULT_TIMEFRAME,
        include_raw_concept=True,
    )

    assert not result.empty
    assert result.iloc[0][COL_RECEIVER_ID] == "R1"
    assert result.iloc[0]["nlp_concepto_sospechoso"] == "Pago extraordinario"
    assert result.iloc[0]["emisores_unicos"] == 3
    assert result.iloc[0]["tx_count"] == 3
    assert result.iloc[0]["conceptos_crudos"] == ["Pago extra", "Pago-Extra"]

    buffer = io.StringIO()
    result.to_csv(buffer, index=False)
    csv_output = buffer.getvalue()
    assert "conceptos_crudos" in csv_output
    assert "Pago extra" in csv_output
