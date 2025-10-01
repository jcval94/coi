
import pandas as pd
from ..schemas import BASE_COLS
from ..utils.time import ensure_time_column, add_month_id

def ingest_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = set(BASE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas base: {missing}")
    df = df.copy()
    df["monto"] = pd.to_numeric(df["monto"], errors="coerce")
    df = ensure_time_column(df, "fecha_hora_ts")
    df = df.dropna(subset=["monto","fecha_hora_ts"])
    df = add_month_id(df, "fecha_hora_ts", "month_id")
    df = df.sort_values("fecha_hora_ts").reset_index(drop=True)
    return df
