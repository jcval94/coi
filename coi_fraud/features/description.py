import re

import pandas as pd

from ..schemas import COL_DESCRIPTION
from ..utils.text import normalize_text


class DescriptionAnalyzer:
    def __init__(self, emo_phrases=None, regex_families=None):
        self.emo = [
            normalize_text(x)
            for x in (emo_phrases or ["gracias mil", "favorzote", "te debo la vida", "eres el mejor"])
        ]
        self.rex = {
            k: re.compile(v)
            for k, v in (
                regex_families
                or {
                    "PRESTAMO": r"\b(?:prestam|adelant|deuda|abono)\b",
                    "VAGUEDAD": r"\b(?:varios|gastos|servicio|pago|consultoria|apoyo|reembolso|proyecto)\b",
                }
            ).items()
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dn = df[COL_DESCRIPTION].fillna("").map(normalize_text)
        vag = []
        emo = []
        flags = []
        for s in dn.tolist():
            toks = s.split()
            vag_hits = len(re.findall(self.rex.get("VAGUEDAD", re.compile("a^")), s))
            v = (vag_hits / max(1, len(toks))) if len(toks) > 2 else max(0.8, vag_hits / max(1, len(toks)))
            vag.append(float(v))
            emo.append(float(sum(1 for p in self.emo if p in s)))
            fams = [fam for fam, rx in self.rex.items() if rx.search(s)]
            flags.append(sorted(set(fams)) if fams else ([] if s else ["SIN_DESCRIPCION"]))
        df["feat_nlp_vaguedad"] = vag
        df["feat_nlp_emocion"] = emo
        df["desc_flags"] = flags
        return df
