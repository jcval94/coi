import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import P
from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_SENDER_ID


ACCENT_MAP = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")


def _normalize_text(value: Optional[str]) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).translate(ACCENT_MAP)
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_datetime(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=True, dayfirst=True)
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(series, errors="coerce", utc=True)
    return dt


def _ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "value_ts" not in frame:
        if "movement_value_date" in frame.columns:
            frame["value_ts"] = _parse_datetime(frame["movement_value_date"])
        elif "fecha_hora_ts" in frame.columns:
            frame["value_ts"] = pd.to_datetime(
                frame["fecha_hora_ts"], errors="coerce", utc=True
            )
        else:
            frame["value_ts"] = pd.NaT
    if "load_ts" not in frame:
        if "load_date" in frame.columns:
            frame["load_ts"] = _parse_datetime(frame["load_date"])
        else:
            frame["load_ts"] = pd.NaT
    frame["value_vs_load_days"] = (
        frame["value_ts"] - frame["load_ts"]
    ).dt.total_seconds().div(86400)
    if "month_id" not in frame:
        value_ts = frame["value_ts"]
        try:
            tzinfo = value_ts.dt.tz
        except AttributeError:
            tzinfo = None
        base = value_ts.dt.tz_convert(None) if tzinfo is not None else value_ts
        frame["month_id"] = base.dt.to_period("M").astype(str)
    return frame


def _collect_manager_cols(df: pd.DataFrame, side_prefix: Optional[str] = None) -> List[str]:
    cols: List[str] = []
    pattern = re.compile(
        rf"^{side_prefix + '-' if side_prefix else ''}manager_\d+_user_id$"
    )
    for column in df.columns:
        if pattern.match(column):
            cols.append(column)
    return cols


def _is_manager_of(row: pd.Series, person_id: str, manager_cols: List[str]) -> bool:
    managers = {str(row[col]) for col in manager_cols if pd.notna(row.get(col))}
    return str(person_id) in managers


def _derive_relation_flags(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    send_id_col = next(
        (c for c in ["envio-user_id", COL_SENDER_ID, "userId", "sender_id"] if c in frame),
        None,
    )
    recv_id_col = next(
        (c for c in ["receptor-user_id", COL_RECEIVER_ID, "receiver_id"] if c in frame),
        None,
    )
    if send_id_col is None or recv_id_col is None:
        frame["is_mgr_chain"] = False
        frame["rel_label"] = "Par/Indefinido"
        return frame

    send_mgr_cols = _collect_manager_cols(frame, side_prefix="envio")
    recv_mgr_cols = _collect_manager_cols(frame, side_prefix="receptor")
    bare_mgr_cols = _collect_manager_cols(frame, side_prefix=None)

    is_chain: List[bool] = []
    rel_labels: List[str] = []
    for _, row in frame.iterrows():
        sender = str(row[send_id_col])
        receiver = str(row[recv_id_col])
        sender_to_receiver = False
        receiver_to_sender = False

        if send_mgr_cols:
            receiver_to_sender = (
                _is_manager_of(row, sender, recv_mgr_cols) if recv_mgr_cols else False
            )
            sender_to_receiver = _is_manager_of(row, receiver, send_mgr_cols)
        elif bare_mgr_cols:
            sender_to_receiver = _is_manager_of(row, receiver, bare_mgr_cols)
            receiver_to_sender = sender == str(row.get("manager_1_user_id")) or sender == str(
                row.get("manager_2_user_id")
            ) or sender == str(row.get("manager_3_user_id")) or sender == str(
                row.get("manager_4_user_id")
            )

        if sender_to_receiver:
            is_chain.append(True)
            rel_labels.append("Subordinado-Manager")
        elif receiver_to_sender:
            is_chain.append(True)
            rel_labels.append("Manager-Subordinado")
        else:
            is_chain.append(False)
            rel_labels.append("Par/Indefinido")

    frame["is_mgr_chain"] = is_chain
    frame["rel_label"] = rel_labels
    return frame


def _zscore_by(df: pd.DataFrame, value_col: str, by_col: str) -> pd.Series:
    groups = df.groupby(by_col, observed=True)[value_col]
    mean = groups.transform("mean")
    std = groups.transform("std").replace(0, np.nan).fillna(1.0)
    return (df[value_col] - mean) / std


def _is_near_threshold(
    value: float, thresholds: Tuple[float, ...], delta: float
) -> Tuple[bool, float, float]:
    if pd.isna(value):
        return (False, np.inf, np.inf)
    distances = [(thr, abs(float(value) - thr)) for thr in thresholds]
    threshold, distance = min(distances, key=lambda item: item[1])
    return (distance <= delta, float(threshold), float(distance))


APPROVAL_RX = re.compile(
    r"(autoriza\w*|aprob\w*|firma\w*|liber\w*|palome\w*|oc\b|po\b|orden de compra|licitaci[oó]n|contrato|alta proveedor)",
    re.IGNORECASE,
)
COMP_RX = re.compile(
    r"(agradec\w*|detalle|incentiv\w*|bono|regalo|mordida|moche|coima|chayote|por fuera|sin factura|sin cfdi|no timbrar|off the record|favor sexual|salida [ií]ntima|cita privada|trato especial)",
    re.IGNORECASE,
)


def _concat_description(row: pd.Series) -> str:
    parts: List[str] = []
    for col in [
        "transaction_desc",
        "reference_number_trans_desc",
        "descripcion",
        "reference_number",
        "concept",
        "tx_detail",
        "tx_service",
    ]:
        if col in row and pd.notna(row[col]):
            text = str(row[col]).strip()
            if text:
                parts.append(text)
    return _normalize_text(" | ".join(parts))


@dataclass
class QuidProQuoConfig:
    window_days: int = 3
    min_score: float = 2.2
    near_thr_delta: float = 10.0
    near_thr: Tuple[float, ...] = tuple(P.near_thresholds)


class QuidProQuoDetector:
    def __init__(self, cfg: QuidProQuoConfig):
        self.cfg = cfg

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            df["feat_quid_score"] = 0.0
            df["sig_quid_pro_quo"] = False
            df["feat_quid_rel_label"] = ""
            df["feat_quid_has_approval"] = False
            df["feat_quid_has_comp"] = False
            df["feat_quid_pair_key"] = ""
            df["feat_quid_desc_norm"] = ""
            df["feat_quid_value_vs_load_days"] = pd.NA
            return df

        work = df.copy()
        work = _ensure_time_columns(work)
        work = _derive_relation_flags(work)

        send_id_col = next(
            (c for c in ["envio-user_id", COL_SENDER_ID, "userId", "sender_id"] if c in work),
            None,
        )
        recv_id_col = next(
            (c for c in ["receptor-user_id", COL_RECEIVER_ID, "receiver_id"] if c in work),
            None,
        )
        if send_id_col is None or recv_id_col is None or COL_AMOUNT not in work:
            df["feat_quid_score"] = 0.0
            df["sig_quid_pro_quo"] = False
            df["feat_quid_rel_label"] = work.get("rel_label", "Par/Indefinido")
            df["feat_quid_has_approval"] = False
            df["feat_quid_has_comp"] = False
            df["feat_quid_pair_key"] = ""
            df["feat_quid_desc_norm"] = ""
            df["feat_quid_value_vs_load_days"] = work.get("value_vs_load_days", pd.NA)
            return df

        work["desc_agg"] = work.apply(_concat_description, axis=1)
        work["has_approval"] = work["desc_agg"].str.contains(APPROVAL_RX, regex=True)
        work["has_comp"] = work["desc_agg"].str.contains(COMP_RX, regex=True)

        work["z_by_sender"] = _zscore_by(work, COL_AMOUNT, by_col=send_id_col)
        near = work[COL_AMOUNT].apply(
            lambda value: _is_near_threshold(
                value, self.cfg.near_thr, self.cfg.near_thr_delta
            )
        )
        work["near_thr_flag"] = near.apply(lambda item: item[0])
        work["near_thr_val"] = near.apply(lambda item: item[1])
        work["near_thr_delta"] = near.apply(lambda item: item[2])

        score = np.zeros(len(work), dtype=float)
        score += work["is_mgr_chain"].astype(float) * 1.0
        score += (work["has_approval"] & work["has_comp"]).astype(float) * 1.2
        score += (work["z_by_sender"] > 2.0).astype(float) * 0.4
        score += work["near_thr_flag"].astype(float) * 0.3
        score += (
            work["value_vs_load_days"].fillna(0) < -0.5
        ).astype(float) * 0.3

        key_ud = work[send_id_col].astype(str) + "|" + work[recv_id_col].astype(str)
        key_du = work[recv_id_col].astype(str) + "|" + work[send_id_col].astype(str)
        work["pair_u_v_undirected"] = np.where(key_ud < key_du, key_ud, key_du)

        work = work.sort_values("value_ts").reset_index()
        idx_col = "index"
        wsecs = int(self.cfg.window_days * 86400)
        boost = np.zeros(len(work), dtype=float)
        for _, indices in work.groupby("pair_u_v_undirected").indices.items():
            idxs = np.array(sorted(indices))
            if len(idxs) < 2:
                continue
            ts_series = work.loc[idxs, "value_ts"]
            valid = ts_series.notna()
            if valid.sum() < 2:
                continue
            ts = ts_series.loc[valid].astype("int64").to_numpy() // 10 ** 9
            appr = work.loc[idxs, "has_approval"].to_numpy()[valid.to_numpy()]
            comp = work.loc[idxs, "has_comp"].to_numpy()[valid.to_numpy()]
            idxs_valid = idxs[valid.to_numpy()]
            i = 0
            for pos, j in enumerate(idxs_valid):
                window_start = ts[pos] - wsecs
                while i < pos and ts[i] < window_start:
                    i += 1
                if appr[pos] and comp[i:pos].any():
                    boost[j] += 0.8
                if comp[pos] and appr[i:pos].any():
                    boost[j] += 0.8
        sorted_score = score[work[idx_col].to_numpy()]
        work["quid_score"] = sorted_score + boost

        reordered = work.set_index(idx_col).reindex(df.index)
        df["feat_quid_score"] = reordered["quid_score"].fillna(0.0).astype(float)
        df["sig_quid_pro_quo"] = df["feat_quid_score"] >= float(self.cfg.min_score)
        df["feat_quid_rel_label"] = reordered["rel_label"].fillna("Par/Indefinido")
        df["feat_quid_has_approval"] = reordered["has_approval"].fillna(False)
        df["feat_quid_has_comp"] = reordered["has_comp"].fillna(False)
        df["feat_quid_pair_key"] = reordered["pair_u_v_undirected"].fillna("")
        df["feat_quid_desc_norm"] = reordered["desc_agg"].fillna("")
        df["feat_quid_value_vs_load_days"] = reordered["value_vs_load_days"]
        return df
