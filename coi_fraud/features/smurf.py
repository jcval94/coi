from collections import defaultdict, deque
from datetime import timedelta

from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_SENDER_ID


class SmurfingDetector:
    def __init__(self, window_days, thresholds):
        self.win = timedelta(days=window_days)
        self.thrs = thresholds

    def transform(self, df):
        df = df.sort_values([COL_SENDER_ID, COL_RECEIVER_ID, "fecha_hora_ts"]).reset_index(drop=True)
        flags = [False] * len(df)
        idxs_by_pair = defaultdict(list)
        for i, (a, b) in enumerate(zip(df[COL_SENDER_ID], df[COL_RECEIVER_ID])):
            idxs_by_pair[(a, b)].append(i)
        for (a, b), idxs in idxs_by_pair.items():
            times = df.loc[idxs, "fecha_hora_ts"].tolist()
            amts = df.loc[idxs, COL_AMOUNT].tolist()
            n = len(idxs)
            i = 0
            sumw = 0.0
            diff = [0] * (n + 1)
            maxdq = deque()
            for j in range(n):
                t = times[j]
                x = amts[j]
                sumw += x
                while maxdq and maxdq[-1][1] <= x:
                    maxdq.pop()
                maxdq.append((j, x))
                while t - times[i] > self.win:
                    sumw -= amts[i]
                    if maxdq and maxdq[0][0] == i:
                        maxdq.popleft()
                    i += 1
                maxv = maxdq[0][1] if maxdq else 0.0
                if any(sumw >= thr and maxv < thr for thr in self.thrs):
                    diff[i] += 1
                    diff[j + 1] -= 1
            acc = 0
            for k in range(n):
                acc += diff[k]
                if acc > 0:
                    flags[idxs[k]] = True
        df["sig_smurf"] = flags
        return df
