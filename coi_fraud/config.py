
from dataclasses import dataclass, field
from typing import Dict, List

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
    near_thresholds: List[float] = field(default_factory=lambda: [500, 1000, 2000, 5000, 10000])
    near_delta: float = 10.0
    round_bases: List[int] = field(default_factory=lambda: [10, 50, 100, 500, 1000])

    weights: Dict[str, float] = field(default_factory=lambda: {
        "zscore": 1.0, "hierarchy": 0.8, "nlp": 0.9, "roundsum": 0.7, "nearthr": 0.9,
        "smurf": 1.3, "yoyo": 1.2, "loan": 1.1, "freq": 0.9, "recurrent": 0.6,
        "sna_cycle": 1.2, "sna_triangle": 0.7
    })

P = Params()
