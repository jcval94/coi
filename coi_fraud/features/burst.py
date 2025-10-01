from dataclasses import dataclass

import pandas as pd

from ..schemas import (
    COL_AMOUNT,
    COL_ORIGIN_APP,
    COL_RECEIVER_ID,
    COL_SENDER_ID,
)


def _format_bin_label(start: pd.Series, end: pd.Series) -> pd.Series:
    start_fmt = start.dt.strftime("%Y-%m-%d %H:%M")
    end_fmt = end.dt.strftime("%H:%M")
    return start_fmt.str.cat(end_fmt, sep=" - ")


@dataclass
class BurstDetectorConfig:
    bin_hours: int
    min_tx: int
    work_start_hour: int
    work_end_hour: int
    min_off_hours_ratio: float


class BurstDetector:
    def __init__(self, config: BurstDetectorConfig):
        self.config = config

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "fecha_hora_ts" not in df:
            return self._ensure_output_columns(df)

        work = df.copy()

        if COL_ORIGIN_APP not in work:
            work[COL_ORIGIN_APP] = pd.NA

        canal = work[COL_ORIGIN_APP].astype("string").str.strip()
        canal = canal.replace({"": pd.NA})
        work[COL_ORIGIN_APP] = canal
        work["rafaga_canal_id"] = canal.fillna("sin_canal")

        ts = pd.to_datetime(work["fecha_hora_ts"], errors="coerce")
        bin_td = pd.Timedelta(hours=self.config.bin_hours)
        work["rafaga_canal_hora_bin_inicio"] = ts.dt.floor(f"{self.config.bin_hours}H")
        work["rafaga_canal_hora_bin_fin"] = work["rafaga_canal_hora_bin_inicio"] + bin_td
        work["rafaga_canal_hora_label"] = _format_bin_label(
            work["rafaga_canal_hora_bin_inicio"], work["rafaga_canal_hora_bin_fin"]
        )

        hour = ts.dt.hour
        off_hours = (hour < self.config.work_start_hour) | (hour >= self.config.work_end_hour)
        work["rafaga_canal_flag_fuera_horario"] = off_hours.fillna(False)

        group_cols = ["rafaga_canal_id", "rafaga_canal_hora_bin_inicio"]
        stats = (
            work.groupby(group_cols, observed=True)
            .agg(
                rafaga_canal_hora_bin_fin=("rafaga_canal_hora_bin_fin", "first"),
                rafaga_canal_hora_label=("rafaga_canal_hora_label", "first"),
                rafaga_canal_tx_en_bin=(COL_AMOUNT, "count"),
                rafaga_canal_tx_fuera_horario=("rafaga_canal_flag_fuera_horario", "sum"),
                rafaga_canal_personas_emisoras=(COL_SENDER_ID, "nunique"),
                rafaga_canal_personas_receptoras=(COL_RECEIVER_ID, "nunique"),
                rafaga_canal_monto_total=(COL_AMOUNT, "sum"),
            )
            .reset_index()
        )

        if not stats.empty:
            stats["rafaga_canal_ratio_fuera_horario"] = (
                stats["rafaga_canal_tx_fuera_horario"]
                / stats["rafaga_canal_tx_en_bin"].replace({0: pd.NA})
            ).fillna(0.0)
            stats["rafaga_canal_flag_evento"] = (
                (stats["rafaga_canal_tx_en_bin"] >= self.config.min_tx)
                & (stats["rafaga_canal_ratio_fuera_horario"] >= self.config.min_off_hours_ratio)
            )
        else:
            stats["rafaga_canal_ratio_fuera_horario"] = pd.Series(dtype="float64")
            stats["rafaga_canal_flag_evento"] = pd.Series(dtype="boolean")

        work = work.merge(stats, on=group_cols, how="left")

        return self._ensure_output_columns(work)

    @staticmethod
    def _ensure_output_columns(df: pd.DataFrame) -> pd.DataFrame:
        required = {
            COL_ORIGIN_APP: "string",
            "rafaga_canal_id": "string",
            "rafaga_canal_hora_bin_inicio": "datetime64[ns]",
            "rafaga_canal_hora_bin_fin": "datetime64[ns]",
            "rafaga_canal_hora_label": "string",
            "rafaga_canal_tx_en_bin": "Int64",
            "rafaga_canal_tx_fuera_horario": "Int64",
            "rafaga_canal_ratio_fuera_horario": "float64",
            "rafaga_canal_personas_emisoras": "Int64",
            "rafaga_canal_personas_receptoras": "Int64",
            "rafaga_canal_monto_total": "float64",
            "rafaga_canal_flag_evento": "boolean",
            "rafaga_canal_flag_fuera_horario": "boolean",
        }
        if df.empty:
            for col, dtype in required.items():
                if col not in df:
                    df[col] = pd.Series(dtype=dtype)
            return df
        for col, dtype in required.items():
            if col not in df:
                df[col] = pd.Series(pd.NA, index=df.index, dtype=dtype)
        return df
