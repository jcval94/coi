
import math
import pandas as pd
from ..config import P
def compute_risk(df: pd.DataFrame) -> pd.DataFrame:
    W = P.weights
    zpos = df["feat_zscore_monto"].astype(float).apply(lambda v: max(v,0.0))

    part_z   = zpos * W["zscore"]
    part_h   = df["relacion"].astype(str).str.contains("Manager", na=False).astype(float) * W["hierarchy"]
    vag = df["feat_nlp_vaguedad"].astype(float)
    emo_flag = (df["feat_nlp_emocion"].astype(float) > 0).astype(float)
    senti = df.get("feat_nlp_sentimiento", 0.0)
    senti = pd.Series(senti, index=df.index).astype(float)
    senti_component = senti.abs() * 0.25
    coi_extra = df.get("feat_nlp_coi_score", 0.0)
    coi_extra = pd.Series(coi_extra, index=df.index).astype(float) * 0.18
    evento_flag = df.get("nlp_evento_corporativo", False)
    evento_flag = pd.Series(evento_flag, index=df.index).astype(float) * 0.2
    part_nlp = (
        (vag + emo_flag * 0.5) * 0.45 * W["nlp"]
        + df["feat_nlp_risk_points"].astype(float) * 0.12
        + senti_component
        + coi_extra
        + evento_flag
    )
    part_rnd = df["sig_roundsum"].astype(float) * W["roundsum"]
    part_thr = df["sig_near_thr"].astype(float) * W["nearthr"]
    part_smf = df["sig_smurf"].astype(float) * W["smurf"]
    part_yy  = df["sig_yoyo"].astype(float) * W["yoyo"]
    part_loan= df["sig_loan_bad_repay"].astype(float) * W["loan"]
    part_freq= df["sig_freq"].astype(float) * W["freq"]
    part_rec = df["sig_recurrent"].astype(float) * W["recurrent"]
    part_cyc = ((df.get("p1_in_cycle",False).astype(bool) | df.get("p2_in_cycle",False).astype(bool)).astype(float)) * W["sna_cycle"]
    part_tri = ((df.get("p1_in_triangle",False).astype(bool) | df.get("p2_in_triangle",False).astype(bool)).astype(float)) * W["sna_triangle"]
    part_quid = df.get("sig_quid_pro_quo", False)
    part_quid = pd.Series(part_quid, index=df.index).astype(float) * W["quid"]
    part_ref = df.get("sig_reference_reuse", False)
    part_ref = pd.Series(part_ref, index=df.index).astype(float) * W["reference_reuse"]
    part_change = df.get("sig_pair_change_point", False)
    part_change = pd.Series(part_change, index=df.index).astype(float) * W["change_point"]
    part_new_edge = df.get("sig_pair_new_edge", False)
    part_new_edge = pd.Series(part_new_edge, index=df.index).astype(float) * W["new_edge"]

    raw = (part_z + part_h + part_nlp + part_rnd + part_thr + part_smf + part_yy +
           part_loan + part_freq + part_rec + part_cyc + part_tri + part_quid +
           part_ref + part_change + part_new_edge)
    df["risk_score"] = raw.apply(lambda x: math.log1p(max(float(x),0.0))*1.3).astype(float)
    return df
