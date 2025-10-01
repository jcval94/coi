import pandas as pd

from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_SENDER_ID


def build_burst_report(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = {
        "rafaga_canal_flag_evento",
        "rafaga_canal_id",
        "rafaga_canal_hora_bin_inicio",
    }
    if df.empty or not required_cols.issubset(df.columns):
        return _empty_report()

    flagged = df[df["rafaga_canal_flag_evento"].fillna(False)].copy()
    if flagged.empty:
        return _empty_report()

    grouped = (
        flagged.groupby(["rafaga_canal_id", "rafaga_canal_hora_bin_inicio"], observed=True)
        .agg(
            rafaga_canal_hora_bin_fin=("rafaga_canal_hora_bin_fin", "first"),
            rafaga_canal_hora_label=("rafaga_canal_hora_label", "first"),
            rafaga_canal_tx_en_bin=("rafaga_canal_tx_en_bin", "first"),
            rafaga_canal_tx_fuera_horario=("rafaga_canal_tx_fuera_horario", "first"),
            rafaga_canal_ratio_fuera_horario=("rafaga_canal_ratio_fuera_horario", "first"),
            rafaga_canal_monto_total=(COL_AMOUNT, "sum"),
            rafaga_canal_monto_promedio=(COL_AMOUNT, "mean"),
            rafaga_canal_personas_emisoras=(COL_SENDER_ID, "nunique"),
            rafaga_canal_personas_receptoras=(COL_RECEIVER_ID, "nunique"),
            rafaga_canal_riesgo_prom=("risk_score", "mean"),
            rafaga_canal_riesgo_max=("risk_score", "max"),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(
        ["rafaga_canal_tx_en_bin", "rafaga_canal_monto_total"], ascending=[False, False]
    )

    return grouped[
        [
            "rafaga_canal_id",
            "rafaga_canal_hora_bin_inicio",
            "rafaga_canal_hora_bin_fin",
            "rafaga_canal_hora_label",
            "rafaga_canal_tx_en_bin",
            "rafaga_canal_tx_fuera_horario",
            "rafaga_canal_ratio_fuera_horario",
            "rafaga_canal_monto_total",
            "rafaga_canal_monto_promedio",
            "rafaga_canal_personas_emisoras",
            "rafaga_canal_personas_receptoras",
            "rafaga_canal_riesgo_prom",
            "rafaga_canal_riesgo_max",
        ]
    ]


def _empty_report() -> pd.DataFrame:
    cols = [
        "rafaga_canal_id",
        "rafaga_canal_hora_bin_inicio",
        "rafaga_canal_hora_bin_fin",
        "rafaga_canal_hora_label",
        "rafaga_canal_tx_en_bin",
        "rafaga_canal_tx_fuera_horario",
        "rafaga_canal_ratio_fuera_horario",
        "rafaga_canal_monto_total",
        "rafaga_canal_monto_promedio",
        "rafaga_canal_personas_emisoras",
        "rafaga_canal_personas_receptoras",
        "rafaga_canal_riesgo_prom",
        "rafaga_canal_riesgo_max",
    ]
    return pd.DataFrame(columns=cols)
