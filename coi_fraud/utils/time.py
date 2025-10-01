
import pandas as pd
TIME_CANDIDATES = ["fecha_hora", "timestamp", "ts", "fecha", "datetime"]

def ensure_time_column(df, out_col="fecha_hora_ts"):
    df = df.copy()
    if out_col in df.columns:
        df[out_col] = pd.to_datetime(df[out_col], errors="coerce", utc=True)
        return df
    src = next((c for c in TIME_CANDIDATES if c in df.columns), None)
    if src is None:
        df[out_col] = pd.NaT
        return df
    s = df[src]
    if pd.api.types.is_numeric_dtype(s):
        med = pd.to_numeric(s, errors="coerce").dropna().median()
        unit = "ms" if pd.notna(med) and med > 1e11 else "s"
        df[out_col] = pd.to_datetime(s, errors="coerce", unit=unit, utc=True)
    else:
        dt = pd.to_datetime(s, errors="coerce", utc=True)
        if dt.isna().mean() > 0.5:
            dt = pd.to_datetime(s, errors="coerce", utc=True, dayfirst=True)
        df[out_col] = dt
    return df

def add_month_id(df, col="fecha_hora_ts", out="month_id"):
    x = df[col].dt.tz_convert(None)
    return df.assign(**{out: x.dt.to_period("M").astype(str)})
