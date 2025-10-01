
import pandas as pd
from ..schemas import TX_COLS_EXPORT
from ..interpret.tx import tx_interpretation
def build_tx_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["interp_tx"] = out.apply(tx_interpretation, axis=1)
    sospechoso = out.get("nlp_concepto_sospechoso", "")
    out["nlp_tx_flag_concepto_sospechoso"] = (
        sospechoso.fillna("").astype("string").str.strip() != ""
    )
    cols = [c for c in TX_COLS_EXPORT if c in out.columns]
    extra_cols = ["nlp_tx_flag_concepto_sospechoso"]
    cols.extend([c for c in extra_cols if c in out.columns and c not in cols])
    out = out[cols].sort_values(["risk_tier","risk_score","month_id"], ascending=[True,False,True])
    cat = pd.Categorical(out["risk_tier"], categories=["CRITICO","ALTO","MEDIO","BAJO"], ordered=True)
    out = out.assign(risk_tier=cat).sort_values(["risk_tier","risk_score"], ascending=[True,False])
    return out
