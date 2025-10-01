
from datetime import timedelta
from ..utils.text import normalize_text
class LoanRepayDetector:
    def __init__(self, days, min_ratio):
        self.days=days; self.min_ratio=min_ratio
    def transform(self, df):
        is_loan = df["descripcion"].fillna("").map(normalize_text).str.contains(r"\b(prestam|adelant|abono)\b", regex=True)
        df["sig_loan_like"]=is_loan
        df["sig_loan_bad_repay"]=False
        df["feat_repay_ratio"]=1.0
        ba_index = {}
        for i,(a,b) in enumerate(zip(df["persona_1"], df["persona_2"])):
            ba_index.setdefault((b,a), []).append(i)
        for i in df.index[df["sig_loan_like"]].tolist():
            a=df.at[i,"persona_1"]; b=df.at[i,"persona_2"]
            t0=df.at[i,"fecha_hora_ts"]; loan=float(df.at[i,"monto"])
            wend = t0 + timedelta(days=self.days)
            repay_sum=0.0
            for j in ba_index.get((a,b),[]):
                t1=df.at[j,"fecha_hora_ts"]
                if t0 <= t1 <= wend: repay_sum += float(df.at[j,"monto"])
            ratio = (repay_sum/loan) if loan>0 else 1.0
            df.at[i,"feat_repay_ratio"]=ratio
            if ratio < self.min_ratio: df.at[i,"sig_loan_bad_repay"]=True
        return df
