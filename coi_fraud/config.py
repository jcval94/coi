
from dataclasses import dataclass, field
from typing import Dict, List


MXN_PER_USD: float = 17.0
MXN_PER_EUR: float = 18.5


def _regulatory_thresholds_mxn() -> List[float]:
    """Regresa los umbrales regulatorios expresados en pesos mexicanos."""

    usd_thresholds = [1000.0 * MXN_PER_USD, 5000.0 * MXN_PER_USD]
    eur_threshold = 5000.0 * MXN_PER_EUR
    values = (*usd_thresholds, eur_threshold)
    return [round(value, 2) for value in sorted(values)]

@dataclass
class Params:
    smurf_window_days: int = 7
    smurf_thresholds: List[float] = field(default_factory=lambda: [10000.0, 20000.0])
    yoyo_hours: int = 8
    yoyo_amount_tol: float = 0.02
    loan_repay_days: int = 60
    loan_min_repay_ratio: float = 0.5
    freq_window_days: int = 30
    freq_pair_threshold: int = 5
    recurrent_months_min: int = 3
    burst_bin_hours: int = 2
    burst_min_tx: int = 5
    burst_work_start_hour: int = 8
    burst_work_end_hour: int = 20
    burst_min_off_hours_ratio: float = 0.6
    near_thresholds: List[float] = field(default_factory=_regulatory_thresholds_mxn)
    near_delta: float = 10.0
    round_bases: List[int] = field(default_factory=lambda: [10, 50, 100, 500, 1000])
    quid_window_days: int = 3
    quid_min_score: float = 2.2
    quid_near_delta: float = 10.0
    ref_reuse_window_days: int = 30
    ref_reuse_min_len: int = 4
    change_point_amount_ratio: float = 4.0
    change_point_count_ratio: float = 3.0
    change_point_min_gap_months: int = 3
    change_point_min_history_months: int = 2

    weights: Dict[str, float] = field(default_factory=lambda: {
        "zscore": 1.0, "hierarchy": 0.8, "nlp": 0.9, "roundsum": 0.7, "nearthr": 0.9,
        "smurf": 1.3, "yoyo": 1.2, "loan": 1.1, "freq": 0.9, "recurrent": 0.6,
        "sna_cycle": 1.2, "sna_triangle": 0.7, "quid": 1.4, "reference_reuse": 1.0,
        "change_point": 1.1, "new_edge": 1.0
    })

P = Params()
