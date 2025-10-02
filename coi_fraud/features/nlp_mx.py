import pandas as pd

from ..schemas import COL_DESCRIPTION, COL_RELATION
from ..text_utils import clean_raw_concept, first_non_empty_series
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
    df2["feat_nlp_sentimiento"] = out["sentimiento"]
    df2["feat_nlp_coi_score"] = out["score_probable_coi"]
    df2["nlp_evento_corporativo"] = out["evento_corporativo"]
    concept_source = first_non_empty_series(
        df2,
        [
            "nlp_concepto_crudo",
            "reference_number_trans_desc",
            COL_DESCRIPTION,
            "tx_tags",
            "feat_reference_norm",
            "nlp_concepto_sospechoso",
        ],
    )
    if concept_source.empty and not df2.empty:
        concept_source = pd.Series([""] * len(df2), index=df2.index, dtype="string")
    df2["nlp_concepto_crudo"] = concept_source.reindex(df2.index, fill_value="").map(clean_raw_concept)
    return df2
