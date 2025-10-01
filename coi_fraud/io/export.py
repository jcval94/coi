
import os
from typing import Dict
import pandas as pd

def export_tables(reports: Dict[str, pd.DataFrame], out_dir: str) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    paths = {}
    for name, df in reports.items():
        p = os.path.join(out_dir, f"{name}_{ts}.csv")
        df.to_csv(p, index=False)
        paths[name] = p
    return paths
