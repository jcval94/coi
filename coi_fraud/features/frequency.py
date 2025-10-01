from datetime import timedelta
from collections import defaultdict

from ..schemas import COL_RECEIVER_ID, COL_SENDER_ID


class FrequencyDetector:
    def __init__(self, window_days, min_cnt):
        self.win = timedelta(days=window_days)
        self.min_cnt = min_cnt

    def transform(self, df):
        df = df.sort_values([COL_SENDER_ID, COL_RECEIVER_ID, "fecha_hora_ts"]).reset_index(drop=True)
        flags = [False] * len(df)
        idxs_by_pair = defaultdict(list)
        for i, (a, b) in enumerate(zip(df[COL_SENDER_ID], df[COL_RECEIVER_ID])):
            idxs_by_pair[(a, b)].append(i)
        for (a, b), idxs in idxs_by_pair.items():
            times = df.loc[idxs, "fecha_hora_ts"].tolist()
            n = len(idxs)
            i = 0
            diff = [0] * (n + 1)
            for j in range(n):
                while times[j] - times[i] > self.win:
                    i += 1
                if (j - i + 1) >= self.min_cnt:
                    diff[i] += 1
                    diff[j + 1] -= 1
            acc = 0
            for k in range(n):
                acc += diff[k]
                if acc > 0:
                    flags[idxs[k]] = True
        df["sig_freq"] = flags
        return df
