
class MonthlyRecurrentDetector:
    def __init__(self, months_min):
        self.min_months = months_min
    def transform(self, df):
        df = df.copy()
        df["dom"] = df["fecha_hora_ts"].dt.day
        dt_naive = df["fecha_hora_ts"].dt.tz_convert(None)
        df["ym"] = dt_naive.dt.to_period("M").astype(str)
        g = (df.groupby(["persona_1","persona_2","dom"], observed=True)["ym"]
               .nunique().reset_index().rename(columns={"ym":"months"}))
        df = df.merge(g, on=["persona_1","persona_2","dom"], how="left")
        df["sig_recurrent"] = df["months"] >= self.min_months
        return df.drop(columns=["dom","ym","months"])
