from ..schemas import COL_AMOUNT


class RoundSumDetector:
    def __init__(self, bases):
        self.bases = bases

    def transform(self, df):
        def is_round(x: float) -> bool:
            for b in self.bases:
                r = x % b
                if r < 1e-9 or abs(b - r) < 1e-9:
                    return True
            return False

        df["sig_roundsum"] = df[COL_AMOUNT].apply(is_round)
        return df
