
import pandas as pd
class Baselines:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby("persona_1", observed=True)["monto"]
        mean = g.transform("mean")
        std = g.transform("std").fillna(1.0)
        df["feat_avg_monto_emisor"] = mean
        df["feat_std_monto_emisor"] = std
        df["feat_zscore_monto"] = (df["monto"] - mean) / std
        return df
