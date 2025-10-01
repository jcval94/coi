
from collections import defaultdict
from datetime import timedelta
class YoYoDetector:
    def __init__(self, hours, tol):
        self.hwin = timedelta(hours=hours); self.tol=tol
    def transform(self, df):
        df = df.sort_values("fecha_hora_ts").reset_index(drop=True)
        flag=[False]*len(df)
        bucket = defaultdict(list)
        for i,(a,b) in enumerate(zip(df["persona_1"], df["persona_2"])):
            key = (a,b) if a<=b else (b,a)
            bucket[key].append(i)
        for key, idxs in bucket.items():
            AtoB=[]; BtoA=[]
            for i in idxs:
                a,b,t,x = df.loc[i,["persona_1","persona_2","fecha_hora_ts","monto"]]
                if key==(a,b): AtoB.append((t,x,i))
                else: BtoA.append((t,x,i))
            AtoB.sort(); BtoA.sort()
            j=0
            for t1,x1,i1 in AtoB:
                while j < len(BtoA) and BtoA[j][0] < t1 - self.hwin: j+=1
                k=j
                while k < len(BtoA) and BtoA[k][0] <= t1 + self.hwin:
                    t2,x2,i2 = BtoA[k]
                    tol_abs = max(x1,x2)*self.tol
                    if abs(x2-x1) <= tol_abs:
                        flag[i1]=True; flag[i2]=True
                    k+=1
        df["sig_yoyo"] = flag
        return df
