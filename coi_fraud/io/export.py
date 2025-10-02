"""Utilities for exporting pipeline tables."""

import os
from typing import Dict

import pandas as pd


def export_tables(
    reports: Dict[str, Dict[str, pd.DataFrame]],
    out_dir: str,
) -> Dict[str, str]:
    """Export pipeline reports grouped by category and timeframe."""

    os.makedirs(out_dir, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
    paths: Dict[str, str] = {}

    for category, periods in reports.items():
        for period, table in periods.items():
            if not isinstance(table, pd.DataFrame):
                raise TypeError(
                    "Expected a pandas DataFrame for report '%s/%s', got %s"
                    % (category, period, type(table).__name__)
                )

            file_name = f"{category}_{period}_{ts}.csv"
            path = os.path.join(out_dir, file_name)
            table.to_csv(path, index=False)
            paths[f"{category}/{period}"] = path

    return paths
