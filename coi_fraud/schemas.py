COL_SENDER_ID = "user_id"
COL_RECEIVER_ID = "receptor-user_id"
COL_RELATION = "relacion"
COL_DATETIME = "fecha_hora"
COL_AMOUNT = "movement_amount"
COL_DESCRIPTION = "descripcion"

COL_SENDER_FULL_NAME = "user_nombre_completo"
COL_RECEIVER_FULL_NAME = "receptor_nombre_completo"
COL_SENDER_JOB = "user_puesto"
COL_RECEIVER_JOB = "receptor_puesto"
COL_SENDER_TENURE_YEARS = "user_antiguedad_anios"
COL_RECEIVER_TENURE_YEARS = "receptor_antiguedad_anios"
COL_SENDER_AGE = "user_edad"
COL_RECEIVER_AGE = "receptor_edad"
COL_SENDER_STATE = "user_estado"
COL_RECEIVER_STATE = "receptor_estado"

BASE_COLS = [
    COL_SENDER_ID,
    COL_RECEIVER_ID,
    COL_RELATION,
    COL_DATETIME,
    COL_AMOUNT,
    COL_DESCRIPTION,
]

PERSON_METADATA_COLS = [
    COL_SENDER_FULL_NAME,
    COL_SENDER_JOB,
    COL_SENDER_TENURE_YEARS,
    COL_SENDER_AGE,
    COL_SENDER_STATE,
    COL_RECEIVER_FULL_NAME,
    COL_RECEIVER_JOB,
    COL_RECEIVER_TENURE_YEARS,
    COL_RECEIVER_AGE,
    COL_RECEIVER_STATE,
]

TX_COLS_EXPORT = [
    "fecha_hora_ts",
    "month_id",
    COL_SENDER_ID,
    COL_RECEIVER_ID,
    COL_RELATION,
    COL_AMOUNT,
    COL_DESCRIPTION,
    COL_SENDER_FULL_NAME,
    COL_SENDER_JOB,
    COL_SENDER_TENURE_YEARS,
    COL_SENDER_AGE,
    COL_SENDER_STATE,
    COL_RECEIVER_FULL_NAME,
    COL_RECEIVER_JOB,
    COL_RECEIVER_TENURE_YEARS,
    COL_RECEIVER_AGE,
    COL_RECEIVER_STATE,
    "risk_score",
    "risk_score_norm",
    "risk_tier",
    "interp_tx",
    "sig_yoyo",
    "sig_smurf",
    "sig_loan_bad_repay",
    "sig_freq",
    "sig_recurrent",
    "sig_roundsum",
    "sig_near_thr",
    "sig_sna_cycle",
    "sig_sna_triangle",
    "sig_quid_pro_quo",
    "feat_quid_score",
    "sig_reference_reuse",
    "feat_reference_norm",
    "nlp_concepto_sospechoso",
    "feat_nlp_risk_points",
    "feat_nlp_vaguedad",
    "feat_nlp_emocion",
]
