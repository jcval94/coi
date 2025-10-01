import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..schemas import COL_RECEIVER_ID, COL_SENDER_ID


ACCENT_MAP = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")


def _normalize_reference(text: Optional[str], normalize_spaces: bool = True) -> str:
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value).translate(ACCENT_MAP)
    value = value.lower().strip()
    if normalize_spaces:
        value = re.sub(r"\s+", " ", value)
    return value


@dataclass
class ReferenceReuseConfig:
    window_days: int = 30
    min_ref_len: int = 4
    normalize_spaces: bool = True


class ReferenceReuseDetector:
    def __init__(self, cfg: ReferenceReuseConfig):
        self.cfg = cfg

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ref_col = "reference_number_trans_desc"
        if ref_col not in df.columns or df.empty:
            df["feat_reference_norm"] = ""
            df["feat_reference_len"] = 0
            df["sig_reference_reuse"] = False
            return df

        normalized = df[ref_col].fillna("").astype(str).map(
            lambda s: _normalize_reference(s, self.cfg.normalize_spaces)
        )
        lengths = normalized.str.len()
        valid_norm = normalized.where(lengths >= int(self.cfg.min_ref_len), "")

        df["feat_reference_norm"] = valid_norm
        df["feat_reference_len"] = lengths

        usable = df[valid_norm != ""].copy()
        if usable.empty:
            df["sig_reference_reuse"] = False
            return df

        send = usable[COL_SENDER_ID].astype(str)
        recv = usable[COL_RECEIVER_ID].astype(str)
        key_ud = send + "|" + recv
        key_du = recv + "|" + send
        usable["pair_undirected"] = np.where(key_ud < key_du, key_ud, key_du)

        usable["fecha_hora_ts"] = pd.to_datetime(
            usable.get("fecha_hora_ts"), errors="coerce", utc=True
        )
        wsecs = int(self.cfg.window_days * 86400)

        reused_refs = set()
        for ref, group in usable.groupby("feat_reference_norm"):
            if group["pair_undirected"].nunique() < 2:
                continue
            times = group["fecha_hora_ts"].dropna().sort_values()
            if times.empty:
                continue
            if (times.max() - times.min()).total_seconds() <= wsecs:
                reused_refs.add(ref)

        df["sig_reference_reuse"] = df["feat_reference_norm"].isin(reused_refs)
        return df
