
from datetime import timedelta
from typing import Optional

import pandas as pd

from .clusters import build_person_clusters
from .concepts import build_concept_tables
from .pairs import build_pair_monthly
from .persons import build_person_monthly
from .transactions import build_tx_table
from .quid import build_quid_tables
from .reference_reuse import build_reference_reuse_tables
from .change_points import build_change_point_tables


TIME_WINDOWS = {
    "ultimo_mes": timedelta(days=30),
    "ultimos_3_meses": timedelta(days=90),
    "todo_el_tiempo": None,
}


def _slice_timeframe(df: pd.DataFrame, window: Optional[timedelta]) -> pd.DataFrame:
    if window is None:
        return df.copy()
    if "fecha_hora_ts" not in df:
        return df.copy()
    max_ts = df["fecha_hora_ts"].max()
    if pd.isna(max_ts):
        return df.iloc[0:0].copy()
    cutoff = max_ts - window
    return df[df["fecha_hora_ts"] >= cutoff].copy()


def _tag_timeframe(table: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(table, pd.DataFrame):
        return table
    tagged = table.copy()
    tagged.insert(0, "timeframe_periodo", label)
    return tagged


def build_all_reports(df: pd.DataFrame):
    outputs = {
        "transaccion": {},
        "persona": {},
        "par_personas": {},
        "concepto_descripcion": {},
        "persona_concepto": {},
        "par_concepto": {},
        "clusters_personas": {},
        "casuistica_quid_pro_quo_tx": {},
        "casuistica_quid_pro_quo_par": {},
        "casuistica_referencia_resumen": {},
        "casuistica_referencia_tx": {},
        "casuistica_cambio_brusco_eventos": {},
        "casuistica_cambio_brusco_pares": {},
    }
    for label, window in TIME_WINDOWS.items():
        sliced = _slice_timeframe(df, window)
        tx = _tag_timeframe(build_tx_table(sliced), label)
        persons = _tag_timeframe(build_person_monthly(sliced), label)
        pairs = _tag_timeframe(build_pair_monthly(sliced), label)
        concepts, persona_concepto, par_concepto = build_concept_tables(sliced)
        concepts = _tag_timeframe(concepts, label)
        persona_concepto = _tag_timeframe(persona_concepto, label)
        par_concepto = _tag_timeframe(par_concepto, label)
        clusters = _tag_timeframe(build_person_clusters(sliced), label)
        quid_tx, quid_pairs = build_quid_tables(sliced)
        quid_tx = _tag_timeframe(quid_tx, label)
        quid_pairs = _tag_timeframe(quid_pairs, label)
        ref_summary, ref_tx = build_reference_reuse_tables(sliced)
        ref_summary = _tag_timeframe(ref_summary, label)
        ref_tx = _tag_timeframe(ref_tx, label)
        chg_events, chg_pairs = build_change_point_tables(sliced)
        chg_events = _tag_timeframe(chg_events, label)
        chg_pairs = _tag_timeframe(chg_pairs, label)
        outputs["transaccion"][label] = tx
        outputs["persona"][label] = persons
        outputs["par_personas"][label] = pairs
        outputs["concepto_descripcion"][label] = concepts
        outputs["persona_concepto"][label] = persona_concepto
        outputs["par_concepto"][label] = par_concepto
        outputs["clusters_personas"][label] = clusters
        outputs["casuistica_quid_pro_quo_tx"][label] = quid_tx
        outputs["casuistica_quid_pro_quo_par"][label] = quid_pairs
        outputs["casuistica_referencia_resumen"][label] = ref_summary
        outputs["casuistica_referencia_tx"][label] = ref_tx
        outputs["casuistica_cambio_brusco_eventos"][label] = chg_events
        outputs["casuistica_cambio_brusco_pares"][label] = chg_pairs
    return outputs
