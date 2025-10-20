from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiment_questions as eq


def _empty_question(*_args, **_kwargs) -> pd.DataFrame:  # type: ignore[override]
    return pd.DataFrame()


def test_question18_handles_alias_columns(monkeypatch):
    for name in (
        "question1_manager_nlp",
        "question2_manager_concepts",
        "question3_quid_pairs",
        "question4_quid_negative_value_vs_load",
        "question5_reference_reuse",
        "question6_centralizers",
        "question7_net_imbalance",
        "question8_case13_new_employees",
        "question9_case14_veterans_from_newcomers",
        "question10_yoyo_streaks",
        "question11_near_threshold_structuring",
        "question12_smurfing_chronic",
        "question13_bad_loans_with_frequency",
        "question14_recurrent_payroll",
        "question15_coordinated_cluster_signals",
        "question16_multisignal_transactions",
        "question17_nlp_person_profiles",
    ):
        monkeypatch.setattr(eq, name, _empty_question)

    timeframe = eq.DEFAULT_TIMEFRAME
    personas = pd.DataFrame(
        {
            "persona_id": ["P001", "P002"],
            "riesgo_promedio_persona": [4.5, 2.1],
            "movimientos_totales": [10, 8],
            "tx_enviadas": [6, 2],
            "monto_enviado": [6000, 2000],
            "tx_recibidas": [4, 6],
            "monto_recibido": [4000, 5000],
            "flujo_neto": [2000, -3000],
            "desbalance_monto_neto": [2000, -3000],
            "desbalance_meses_totales": [5, 4],
            "tasa_meses_envia_extremo": [0.2, 0.0],
            "tasa_meses_recibe_extremo": [0.0, 0.25],
            "banderas_activas": [2, 1],
            "tasa_flag_maxima": [0.3, 0.1],
            "banderas_principales": ["bandera A", "bandera B"],
            "casuistica_puntaje_total": [5.0, 3.0],
            "casuistica_puntaje_promedio": [1.0, 0.6],
            "casuistica_resumen_texto": ["Detalle A", "Detalle B"],
        }
    )

    transacciones = pd.DataFrame(
        {
            "destinatario_id": ["P001", "P001", "P002"],
            "monto_movimiento": [1000, 2000, 1500],
            "fecha_operacion": pd.to_datetime(
                ["2024-01-05", "2024-01-20", "2024-02-10"]
            ),
            "mes": ["2024-01", "2024-01", "2024-02"],
        }
    )

    reports = {
        "persona": {timeframe: personas},
        "transaccion": {timeframe: transacciones},
    }

    result = eq.question18_user_risk_scores(reports, timeframe=timeframe, top_n=2)

    assert not result.empty
    assert set(result["persona"]) == {"P001", "P002"}
    persona_p1 = result.loc[result["persona"] == "P001"].iloc[0]
    assert persona_p1["risk_avg_person"] == 4.5
    assert {
        "persona",
        "risk_avg_person",
        "n_tx_emit",
        "n_tx_recv",
        "sum_emit",
        "sum_recv",
        "banderas_destacadas",
        "bandera_manager_nlp",
        "casuistica_score_total_todas_temporalidades",
    }.issubset(result.columns)
    assert (
        persona_p1["casuistica_score_total_todas_temporalidades"]
        == persona_p1["casuistica_score_total"]
    )
    assert persona_p1["bandera_manager_nlp"] == "SIN_ALERTA"
    detalle = persona_p1["detalle_pagos_mensuales"]
    assert isinstance(detalle, list) and len(detalle) == 1
