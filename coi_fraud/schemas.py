
BASE_COLS = ["persona_1","persona_2","relacion","fecha_hora","monto","descripcion"]

TX_COLS_EXPORT = [
    "fecha_hora_ts","month_id","persona_1","persona_2","relacion","monto","descripcion",
    "risk_score","risk_score_norm","risk_tier","interp_tx",
    "sig_yoyo","sig_smurf","sig_loan_bad_repay","sig_freq","sig_recurrent",
    "sig_roundsum","sig_near_thr","sig_sna_cycle","sig_sna_triangle",
    "nlp_concepto_sospechoso","feat_nlp_risk_points","feat_nlp_vaguedad","feat_nlp_emocion"
]
