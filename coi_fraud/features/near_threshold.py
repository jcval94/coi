
class NearThresholdDetector:
    def __init__(self, thresholds, delta):
        self.thrs = thresholds; self.delta = delta
    def transform(self, df):
        def f(x):
            ds=[abs(x-t) for t in self.thrs]; m=min(ds) if ds else 1e9
            return (m<=self.delta, m)
        vals = df["monto"].apply(f)
        df["sig_near_thr"] = vals.apply(lambda v: v[0])
        df["feat_delta_near_thr"] = vals.apply(lambda v: v[1])
        return df
