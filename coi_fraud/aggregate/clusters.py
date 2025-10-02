from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set

import pandas as pd

from ..schemas import COL_AMOUNT, COL_DESCRIPTION, COL_RECEIVER_ID, COL_SENDER_ID


def _build_components(edges: Iterable[tuple]) -> List[Set]:
    adjacency: Dict = defaultdict(set)
    nodes: Set = set()
    for sender, receiver in edges:
        if pd.isna(sender) or pd.isna(receiver):
            continue
        adjacency[sender].add(receiver)
        adjacency[receiver].add(sender)
        nodes.add(sender)
        nodes.add(receiver)
    visited: Set = set()
    components: List[Set] = []
    for node in sorted(nodes):
        if node in visited:
            continue
        queue = deque([node])
        comp: Set = set()
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            comp.add(current)
            neighbours = sorted(adjacency.get(current, set()))
            for neighbour in neighbours:
                if neighbour not in visited:
                    queue.append(neighbour)
        if comp:
            components.append(comp)
    return components


def _collect_texts(series: pd.Series) -> List[str]:
    if series is None or series.empty:
        return []
    texts = series.astype("string").fillna("").str.strip()
    return [text for text in texts.tolist() if text]


def _top_concepts(df: pd.DataFrame) -> tuple[List[str], List[List[str]]]:
    if df.empty:
        return [], []
    work = df.copy()
    text_series = work.get(COL_DESCRIPTION)
    if text_series is None:
        text_series = pd.Series("", index=work.index, dtype="string")
    else:
        text_series = text_series.astype("string").fillna("").str.strip()
    work["_nlp_texto_original"] = text_series
    counts = (
        work.groupby("nlp_concepto_sospechoso", observed=True)
        .agg(
            cnt=(COL_AMOUNT, "count"),
            textos=("_nlp_texto_original", _collect_texts),
        )
        .reset_index()
        .sort_values(["cnt", "nlp_concepto_sospechoso"], ascending=[False, True])
    )
    top = counts.head(3)
    return top["nlp_concepto_sospechoso"].tolist(), top["textos"].tolist()


def build_person_clusters(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_personas",
                "cluster_personas_total",
                "cluster_tx_count",
                "cluster_tx_sum",
                "riesgo_cluster_maximo",
                "riesgo_cluster_promedio",
                "nlp_cluster_total_transacciones_sospechosas",
                "nlp_cluster_conceptos_sospechosos_unicos",
                "nlp_cluster_top_conceptos",
                "nlp_cluster_top_textos",
                "yo_yo_cluster_tasa_flag",
                "smurf_cluster_tasa_flag",
                "frecuencia_cluster_tasa_flag",
                "recurrente_cluster_tasa_flag",
                "prestamo_cluster_tasa_repay_insuficiente",
                "monto_cluster_tasa_flag_redondo",
                "umbral_cluster_tasa_flag_cercania",
                "red_cluster_tasa_en_ciclos",
                "red_cluster_tasa_en_triangulos",
                "quid_cluster_tasa_flag",
                "referencia_cluster_tasa_reutilizada",
                "desbalance_cluster_persona_principal",
                "desbalance_cluster_persona_principal_monto",
                "interp_cluster",
            ]
        )

    df = df.copy()
    for col in ["sig_quid_pro_quo", "sig_reference_reuse"]:
        if col not in df:
            df[col] = 0.0

    components = _build_components(zip(df[COL_SENDER_ID], df[COL_RECEIVER_ID]))
    records: List[Dict] = []
    concept_series = df.get("nlp_concepto_sospechoso", pd.Series([], dtype="string"))
    concept_series = concept_series.fillna("").astype("string").str.strip()

    for idx, comp in enumerate(components, start=1):
        cluster_df = df[df[COL_SENDER_ID].isin(comp) & df[COL_RECEIVER_ID].isin(comp)].copy()
        if cluster_df.empty:
            continue
        cluster_concepts = concept_series.loc[cluster_df.index]
        suspicious_df = cluster_df.loc[cluster_concepts != ""]
        emit = cluster_df.groupby(COL_SENDER_ID, observed=True)[COL_AMOUNT].sum()
        recv = cluster_df.groupby(COL_RECEIVER_ID, observed=True)[COL_AMOUNT].sum()
        balance = emit.sub(recv, fill_value=0.0)
        if not balance.empty:
            principal = balance.reindex(balance.abs().sort_values(ascending=False).index)
            desbalance_persona = principal.index[0]
            desbalance_monto = float(principal.iloc[0])
        else:
            desbalance_persona = None
            desbalance_monto = 0.0

        riesgo_max = cluster_df["risk_score"].max()
        riesgo_avg = cluster_df["risk_score"].mean()
        riesgo_max = float(riesgo_max) if pd.notna(riesgo_max) else 0.0
        riesgo_avg = float(riesgo_avg) if pd.notna(riesgo_avg) else 0.0
        top_concepts, top_texts = _top_concepts(suspicious_df)

        record = {
            "cluster_id": f"cluster_{idx}",
            "cluster_personas": sorted(comp),
            "cluster_personas_total": len(comp),
            "cluster_tx_count": int(cluster_df.shape[0]),
            "cluster_tx_sum": float(cluster_df[COL_AMOUNT].sum()),
            "riesgo_cluster_maximo": riesgo_max,
            "riesgo_cluster_promedio": riesgo_avg,
            "nlp_cluster_total_transacciones_sospechosas": int(suspicious_df.shape[0]),
            "nlp_cluster_conceptos_sospechosos_unicos": int(
                suspicious_df["nlp_concepto_sospechoso"].nunique()
            ),
            "nlp_cluster_top_conceptos": top_concepts,
            "nlp_cluster_top_textos": top_texts,
            "yo_yo_cluster_tasa_flag": float(cluster_df["sig_yoyo"].fillna(0.0).mean()),
            "smurf_cluster_tasa_flag": float(cluster_df["sig_smurf"].fillna(0.0).mean()),
            "frecuencia_cluster_tasa_flag": float(cluster_df["sig_freq"].fillna(0.0).mean()),
            "recurrente_cluster_tasa_flag": float(
                cluster_df["sig_recurrent"].fillna(0.0).mean()
            ),
            "prestamo_cluster_tasa_repay_insuficiente": float(
                cluster_df.get(
                    "sig_loan_bad_repay", pd.Series(0.0, index=cluster_df.index)
                )
                .fillna(0.0)
                .mean()
            ),
            "monto_cluster_tasa_flag_redondo": float(
                cluster_df.get(
                    "sig_roundsum", pd.Series(0.0, index=cluster_df.index)
                )
                .fillna(0.0)
                .mean()
            ),
            "umbral_cluster_tasa_flag_cercania": float(
                cluster_df.get("sig_near_thr", pd.Series(0.0, index=cluster_df.index))
                .fillna(0.0)
                .mean()
            ),
            "red_cluster_tasa_en_ciclos": float(
                cluster_df.get("p1_in_cycle", pd.Series(0.0, index=cluster_df.index))
                .fillna(0.0)
                .mean()
            ),
            "red_cluster_tasa_en_triangulos": float(
                cluster_df.get(
                    "p1_in_triangle", pd.Series(0.0, index=cluster_df.index)
                )
                .fillna(0.0)
                .mean()
            ),
            "quid_cluster_tasa_flag": float(
                cluster_df.get(
                    "sig_quid_pro_quo", pd.Series(0.0, index=cluster_df.index)
                )
                .fillna(0.0)
                .mean()
            ),
            "referencia_cluster_tasa_reutilizada": float(
                cluster_df.get(
                    "sig_reference_reuse", pd.Series(0.0, index=cluster_df.index)
                )
                .fillna(0.0)
                .mean()
            ),
            "desbalance_cluster_persona_principal": desbalance_persona,
            "desbalance_cluster_persona_principal_monto": desbalance_monto,
        }
        record["interp_cluster"] = (
            f"{record['cluster_personas_total']} personas con {record['cluster_tx_count']} tx "
            f"(${record['cluster_tx_sum']:,.0f}). Riesgo max {record['riesgo_cluster_maximo']:.2f}."
        )
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_personas",
                "cluster_personas_total",
                "cluster_tx_count",
                "cluster_tx_sum",
                "riesgo_cluster_maximo",
                "riesgo_cluster_promedio",
                "nlp_cluster_total_transacciones_sospechosas",
                "nlp_cluster_conceptos_sospechosos_unicos",
                "nlp_cluster_top_conceptos",
                "yo_yo_cluster_tasa_flag",
                "smurf_cluster_tasa_flag",
                "frecuencia_cluster_tasa_flag",
                "recurrente_cluster_tasa_flag",
                "prestamo_cluster_tasa_repay_insuficiente",
                "monto_cluster_tasa_flag_redondo",
                "umbral_cluster_tasa_flag_cercania",
                "red_cluster_tasa_en_ciclos",
                "red_cluster_tasa_en_triangulos",
                "desbalance_cluster_persona_principal",
                "desbalance_cluster_persona_principal_monto",
                "interp_cluster",
            ]
        )

    return pd.DataFrame(records).sort_values(
        ["riesgo_cluster_maximo", "cluster_tx_sum"], ascending=[False, False]
    )
