import re
from typing import Iterable, Optional, Sequence

import pandas as pd

from ..schemas import (
    BASE_COLS,
    COL_AMOUNT,
    COL_DATETIME,
    COL_DESCRIPTION,
    COL_RECEIVER_AGE,
    COL_RECEIVER_FULL_NAME,
    COL_RECEIVER_JOB,
    COL_RECEIVER_STATE,
    COL_RECEIVER_TENURE_YEARS,
    COL_RECEIVER_ID,
    COL_RELATION,
    COL_SENDER_AGE,
    COL_SENDER_FULL_NAME,
    COL_SENDER_JOB,
    COL_SENDER_STATE,
    COL_SENDER_TENURE_YEARS,
    COL_SENDER_ID,
)
from ..utils.time import ensure_time_column, add_month_id
from ..utils.tx_desc import split_transaction_desc

LOAD_DATE_COL = "load_date"
TX_DESC_COL = "transaction_desc"
TEAMMATES_COL = "companeros_de_equipo"
MANAGER_COLS = [f"manager_{i}_user_id" for i in range(1, 5)]
SENDER_META_SOURCE = {
    "envio-nombre_completo": COL_SENDER_FULL_NAME,
    "envio-puesto": COL_SENDER_JOB,
    "envio-edad": COL_SENDER_AGE,
    "envio-state_id": COL_SENDER_STATE,
}
RECEIVER_META_SOURCE = {
    "receptor-nombre_completo": COL_RECEIVER_FULL_NAME,
    "receptor-puesto": COL_RECEIVER_JOB,
    "receptor-edad": COL_RECEIVER_AGE,
    "receptor-state_id": COL_RECEIVER_STATE,
}
SENDER_HIRE_DATE_COL = "envio-gf_worker_hiring_date"
RECEIVER_HIRE_DATE_COL = "receptor-gf_worker_hiring_date"


def _normalize_ampm(text: str) -> str:
    text = text.strip()
    # Unifica variaciones en español de AM/PM
    text = re.sub(r"(?i)\b([ap])\.?\s*m\.?\b", lambda m: "AM" if m.group(1).lower() == "a" else "PM", text)
    return text


def _parse_load_datetime(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").fillna("").map(lambda x: _normalize_ampm(x) if x else x)
    dt = pd.to_datetime(cleaned, errors="coerce")
    return dt.mask(cleaned == "")


def _split_teammates(value: Optional[str]) -> Sequence[str]:
    if pd.isna(value):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if pd.notna(v) and str(v).strip()]
    text = str(value)
    if not text:
        return []
    parts = re.split(r"[\s,;|_]+", text)
    return [p for p in (part.strip() for part in parts) if p]


def _infer_relacion(row: pd.Series) -> str:
    sender = row.get(COL_SENDER_ID)
    receiver = row.get(COL_RECEIVER_ID)
    if pd.isna(receiver) or receiver == "":
        return "sin_dato"
    if receiver == sender:
        return "mismo_usuario"
    managers = {row.get(col) for col in MANAGER_COLS if col in row and pd.notna(row[col])}
    if receiver in managers:
        return "manager_del_emisor"
    teammates = set(_split_teammates(row.get(TEAMMATES_COL)))
    if receiver in teammates:
        return "companero_equipo"
    return "otro"


def _compute_tenure_years(reference_dt: pd.Series, hire_series: Optional[pd.Series]) -> pd.Series:
    if hire_series is None:
        return pd.Series(pd.NA, index=reference_dt.index, dtype="float64")
    ref = pd.to_datetime(reference_dt, errors="coerce")
    hire = pd.to_datetime(hire_series, errors="coerce", dayfirst=True)
    try:
        tz = ref.dt.tz
    except AttributeError:
        tz = None
    if tz is not None:
        ref = ref.dt.tz_convert(None)
    try:
        tz_hire = hire.dt.tz
    except AttributeError:
        tz_hire = None
    if tz_hire is not None:
        hire = hire.dt.tz_convert(None)
    tenure_days = (ref - hire).dt.days
    tenure_years = tenure_days / 365.25
    return tenure_years.round(2)


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas base: {missing}")


def ingest_df(df: pd.DataFrame) -> pd.DataFrame:
    raw_required = {COL_SENDER_ID, COL_RECEIVER_ID, COL_AMOUNT, LOAD_DATE_COL, TX_DESC_COL}
    _ensure_columns(df, raw_required)
    df = df.copy()

    origin_col = "origin_application_id"
    if origin_col in df.columns:
        df[origin_col] = df[origin_col].astype("string").str.strip()
        df[origin_col] = df[origin_col].replace({"": pd.NA})
    else:
        df[origin_col] = pd.NA

    df[COL_AMOUNT] = pd.to_numeric(df[COL_AMOUNT], errors="coerce")
    df[COL_DATETIME] = _parse_load_datetime(df[LOAD_DATE_COL])

    df = split_transaction_desc(df, col=TX_DESC_COL, prefix="tx_", keep_original=True)
    service = df["tx_service"].astype("string").fillna("").str.strip()
    detail = df["tx_detail"].astype("string").fillna("").str.strip()
    code = df["tx_code"].astype("string").fillna("").str.strip()
    combined = service.str.cat(detail.where(detail != ""), sep=" ", na_rep="").str.replace(r"\s+", " ", regex=True).str.strip()
    combined = combined.str.cat(code.where(code != ""), sep=" ", na_rep="").str.replace(r"\s+", " ", regex=True).str.strip()
    fallback = df[TX_DESC_COL].astype("string").fillna("").str.strip()
    desc = combined.where(combined != "", fallback)
    df[COL_DESCRIPTION] = desc.replace({"": pd.NA})

    df[COL_RELATION] = df.apply(_infer_relacion, axis=1)

    for src, dst in SENDER_META_SOURCE.items():
        df[dst] = df[src] if src in df.columns else pd.NA
    for src, dst in RECEIVER_META_SOURCE.items():
        df[dst] = df[src] if src in df.columns else pd.NA

    df[COL_SENDER_TENURE_YEARS] = _compute_tenure_years(df[COL_DATETIME], df.get(SENDER_HIRE_DATE_COL))
    df[COL_RECEIVER_TENURE_YEARS] = _compute_tenure_years(df[COL_DATETIME], df.get(RECEIVER_HIRE_DATE_COL))

    df[COL_SENDER_AGE] = pd.to_numeric(df[COL_SENDER_AGE], errors="coerce")
    df[COL_RECEIVER_AGE] = pd.to_numeric(df[COL_RECEIVER_AGE], errors="coerce")

    _ensure_columns(df, BASE_COLS)
    df = ensure_time_column(df, "fecha_hora_ts")
    df = df.dropna(subset=[COL_AMOUNT, "fecha_hora_ts"])
    df = add_month_id(df, "fecha_hora_ts", "month_id")
    df = df.sort_values("fecha_hora_ts").reset_index(drop=True)

    return df
