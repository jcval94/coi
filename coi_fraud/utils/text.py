
import re
ACCENT_MAP = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
def normalize_text(s):
    if s is None: return ""
    s2 = str(s).strip().lower().translate(ACCENT_MAP)
    return re.sub(r"\s+", " ", s2)
