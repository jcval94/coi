
import pandas as pd
from datetime import date, timedelta

def _easter_date(year: int) -> date:
    a = year % 19; b = year // 100; c = year % 100
    d = b // 4; e = b % 4; f = (b + 8) // 25; g = (b - f + 1) // 3
    h = (19*a + b - d - g + 15) % 30
    i = c // 4; k = c % 4
    l = (32 + 2*e + 2*i - h - k) % 7
    m = (a + 11*h + 22*l) // 451
    month = (h + l - 7*m + 114) // 31
    day = ((h + l - 7*m + 114) % 31) + 1
    return date(year, month, day)

def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    while d.weekday() != weekday:
        d += timedelta(days=1)
    return d + timedelta(days=7*(n-1))

def _third_monday(year: int, month: int) -> date:
    return _nth_weekday_of_month(year, month, weekday=0, n=3)

def _first_monday(year: int, month: int) -> date:
    return _nth_weekday_of_month(year, month, weekday=0, n=1)

def mexican_holidays(year: int, include_cultural: bool = True):
    easter = _easter_date(year); good_friday = easter - timedelta(days=2)
    d = {
        date(year, 1, 1): "Año Nuevo",
        _first_monday(year, 2): "Constitución (1er lunes feb)",
        _third_monday(year, 3): "Natalicio Benito Juárez (3er lunes mar)",
        date(year, 5, 1): "Día de Trabajo",
        date(year, 9, 16): "Independencia",
        _third_monday(year, 11): "Revolución Mexicana (3er lunes nov)",
        date(year, 12, 25): "Navidad",
        good_friday: "Viernes Santo",
    }
    if include_cultural:
        d[date(year, 11, 2)] = "Día de Muertos (cultural)"
    return d

def holiday_proximity(df, window_days=3, attenuation=0.5, include_cultural=True):
    x = df["fecha_hora_ts"].dt.tz_convert(None).dt.date
    years = sorted(set(d.year for d in x))
    hol = {}
    for y in years:
        hol.update(mexican_holidays(y, include_cultural=include_cultural))
    hol_dates = sorted(list(hol.keys()))
    hol_names = {d: name for d,name in hol.items()}

    import bisect
    def nearest(d):
        i = bisect.bisect_left(hol_dates, d)
        best = None; candidates = []
        if i < len(hol_dates): candidates.append(hol_dates[i])
        if i > 0: candidates.append(hol_dates[i-1])
        for hd in candidates:
            dist = abs((hd - d).days)
            if best is None or dist < best[1]:
                best = (hol_names[hd], dist)
        return best if best else ("(sin feriado)", 9999)

    res = x.map(nearest)
    names = [r[0] for r in res]; days = [r[1] for r in res]
    out = df.copy()
    out["hol_name"] = names
    out["hol_days"] = days
    out["hol_within"] = out["hol_days"] <= window_days
    out["hol_attenuation"] = out["hol_within"].map({True: attenuation, False: 1.0})
    return out
