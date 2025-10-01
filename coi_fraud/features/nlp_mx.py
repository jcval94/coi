import pandas as pd

from ..schemas import COL_DESCRIPTION, COL_RELATION
from .nlp_mx_impl import nlp_mx_etiquetar_transacciones_pro


def apply_nlp(df: pd.DataFrame) -> pd.DataFrame:
    out = nlp_mx_etiquetar_transacciones_pro(
        df,
        col_texto=COL_DESCRIPTION,
        col_relacion=COL_RELATION,
        use_embeddings=True,
    )
    df2 = df.copy()
    df2["nlp_concepto_sospechoso"] = out["concepto sospechoso"]
    df2["feat_nlp_risk_points"] = out["riesgo_puntos"]
    df2["feat_nlp_vaguedad"] = out["vaguedad"]
    df2["feat_nlp_emocion"] = out["emocion"]
    return df2
