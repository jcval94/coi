import pandas as pd

from ..schemas import COL_AMOUNT, COL_SENDER_ID


class Baselines:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby(COL_SENDER_ID, observed=True)[COL_AMOUNT]
        mean = g.transform("mean")
        std = g.transform("std").fillna(1.0)
        df["feat_avg_monto_emisor"] = mean
        df["feat_std_monto_emisor"] = std
        df["feat_zscore_monto"] = (df[COL_AMOUNT] - mean) / std
        return df
