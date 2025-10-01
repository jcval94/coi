
import pandas as pd
from .io.ingest import ingest_df
from .features.baselines import Baselines
from .features.description import DescriptionAnalyzer
from .features.roundsum import RoundSumDetector
from .features.near_threshold import NearThresholdDetector
from .features.smurf import SmurfingDetector
from .features.yoyo import YoYoDetector
from .features.loan import LoanRepayDetector
from .features.frequency import FrequencyDetector
from .features.recurrent import MonthlyRecurrentDetector
from .features.quid import QuidProQuoConfig, QuidProQuoDetector
from .features.reference_reuse import ReferenceReuseConfig, ReferenceReuseDetector
from .features.change_points import ChangePointConfig, ChangePointDetector
from .features.sna_light import transform as sna_light
from .features.holidays import holiday_proximity
from .features.nlp_mx import apply_nlp
from .scoring.engine import compute_risk
from .scoring.calibrate import monthly_percentiles, add_norm_and_tier
from .aggregate.reports import build_all_reports
from .config import P

def run_pipeline(df, language="es"):
    df = ingest_df(df)
    for step in [
        Baselines(), DescriptionAnalyzer(),
        RoundSumDetector(P.round_bases), NearThresholdDetector(P.near_thresholds, P.near_delta),
        SmurfingDetector(P.smurf_window_days, P.smurf_thresholds),
        YoYoDetector(P.yoyo_hours, P.yoyo_amount_tol),
        LoanRepayDetector(P.loan_repay_days, P.loan_min_repay_ratio),
        FrequencyDetector(P.freq_window_days, P.freq_pair_threshold),
        MonthlyRecurrentDetector(P.recurrent_months_min),
        QuidProQuoDetector(
            QuidProQuoConfig(
                window_days=P.quid_window_days,
                min_score=P.quid_min_score,
                near_thr_delta=P.quid_near_delta,
                near_thr=tuple(P.near_thresholds),
            )
        ),
        ReferenceReuseDetector(
            ReferenceReuseConfig(
                window_days=P.ref_reuse_window_days,
                min_ref_len=P.ref_reuse_min_len,
            )
        ),
        ChangePointDetector(
            ChangePointConfig(
                monthly_amount_ratio=P.change_point_amount_ratio,
                monthly_count_ratio=P.change_point_count_ratio,
                min_gap_months=P.change_point_min_gap_months,
                min_history_months=P.change_point_min_history_months,
            )
        ),
    ]:
        df = step.transform(df)
    df = sna_light(df)
    df = apply_nlp(df)
    hol = holiday_proximity(df)
    df["hol_attenuation"] = hol["hol_attenuation"]
    df = compute_risk(df)
    calib = monthly_percentiles(df, score_col="risk_score", month_col="month_id")
    df = add_norm_and_tier(df, calib, score_col="risk_score", month_col="month_id")
    return build_all_reports(df)
