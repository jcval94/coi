import re

import pandas as pd

from ..schemas import (
    COL_AMOUNT,
    COL_DESCRIPTION,
    COL_RECEIVER_ID,
    COL_RELATION,
    COL_SENDER_ID,
)


def _get_section(reports, section, timeframe="todo_el_tiempo"):
    data = reports.get(section, {})
    if isinstance(data, dict):
        df = data.get(timeframe)
    else:
        df = data
    if isinstance(df, pd.DataFrame):
        out = df.copy()
        if "timeframe_periodo" in out.columns:
            out = out.drop(columns=["timeframe_periodo"])
        return out
    return pd.DataFrame()


def _get_tx(reports, timeframe="todo_el_tiempo"):
    return _get_section(reports, "transaccion", timeframe)


def empleados_recepcion_constante(reports, min_meses=3, timeframe="todo_el_tiempo"):
    tx = _get_tx(reports, timeframe)
    if tx.empty or "month_id" not in tx:
        return pd.DataFrame(columns=["pair", "meses_con_tx", "meses_recurrente"])
    pairs = tx.copy()
    pairs["pair"] = pairs[COL_SENDER_ID].astype(str) + "→" + pairs[COL_RECEIVER_ID].astype(str)
    monthly = (
        pairs.groupby(["month_id", "pair"], observed=True)
        .agg(
            tx_count=(COL_AMOUNT, "count"),
            pct_recurrent=("sig_recurrent", "mean"),
        )
        .reset_index()
    )
    out = (
        monthly.assign(tiene_tx=monthly["tx_count"] > 0, tiene_rec=monthly["pct_recurrent"] > 0)
        .groupby("pair", as_index=False)
        .agg(
            meses_con_tx=("tiene_tx", "sum"),
            meses_recurrente=("tiene_rec", "sum"),
        )
        .query("meses_con_tx >= @min_meses")
        .sort_values(["meses_recurrente", "meses_con_tx"], ascending=[False, False])
    )
    return out


def desbalance_personas(reports, timeframe="todo_el_tiempo"):
    pers = _get_section(reports, "persona", timeframe)
    if pers.empty:
        return pers
    desbalance = pers.assign(
        net=pers["sum_emit"] - pers["sum_recv"],
        ratio=(pers["sum_emit"] / (pers["sum_recv"] + 1e-9)),
    ).sort_values(["net"], ascending=False)
    return desbalance


def manager_conceptos_sospechosos(reports, keywords=None, timeframe="todo_el_tiempo"):
    tx = _get_tx(reports, timeframe)
    if tx.empty:
        return tx
    mgr = tx[tx[COL_RELATION].str.contains("Manager", case=False, na=False)].copy()
    if keywords:
        pat = re.compile("|".join([re.escape(k) for k in keywords]), re.IGNORECASE)
        mgr = mgr[mgr[COL_DESCRIPTION].fillna("").str.contains(pat)]
    agg = (
        mgr.groupby(["month_id", "nlp_concepto_sospechoso"], as_index=False)
        .agg(
            tx_count=("risk_score", "count"),
            risk_p95=("risk_score", lambda s: float(s.quantile(0.95)) if len(s) else 0.0),
        )
        .sort_values(["month_id", "risk_p95", "tx_count"], ascending=[True, False, False])
    )
    return agg


def centralizadores(reports, timeframe="todo_el_tiempo"):
    tx = _get_tx(reports, timeframe)[
        ["month_id", COL_SENDER_ID, COL_RECEIVER_ID, COL_AMOUNT, "risk_score"]
    ].copy()
    if tx.empty:
        return tx
    centro = (
        tx.groupby(["month_id", COL_RECEIVER_ID], as_index=False)
        .agg(
            inflow=(COL_AMOUNT, "sum"),
            emisores_unicos=(COL_SENDER_ID, "nunique"),
            n_tx=(COL_AMOUNT, "count"),
            risk_avg=("risk_score", "mean"),
        )
        .assign(centralidad=lambda d: d["inflow"] * d["emisores_unicos"])
        .sort_values(["month_id", "centralidad"], ascending=[True, False])
    )
    return centro


def yoyo_consecutivos(reports, timeframe="todo_el_tiempo"):
    tx = _get_tx(reports, timeframe)
    if tx.empty or "month_id" not in tx:
        return pd.DataFrame(columns=["pair", "tiene_racha_yo_yo"])
    tmp = tx.copy()
    tmp["pair"] = tmp[COL_SENDER_ID].astype(str) + "→" + tmp[COL_RECEIVER_ID].astype(str)
    monthly = (
        tmp.groupby(["month_id", "pair"], observed=True)["sig_yoyo"].mean().reset_index(name="pct_yoyo")
    )
    monthly["hit"] = monthly["pct_yoyo"] > 0
    out = (
        monthly.sort_values(["pair", "month_id"])
        .groupby("pair")["hit"]
        .apply(lambda s: any(s.shift(1).fillna(False) & s))
        .reset_index(name="tiene_racha_yo_yo")
    )
    return out[out["tiene_racha_yo_yo"]]


def near_thr_repetido(reports, min_meses=2, timeframe="todo_el_tiempo"):
    tx = _get_tx(reports, timeframe)
    if tx.empty or "month_id" not in tx:
        return pd.DataFrame(columns=["pair", "meses_con_near"])
    tmp = tx[["month_id", COL_SENDER_ID, COL_RECEIVER_ID, "sig_near_thr"]].copy()
    tmp["pair"] = tmp[COL_SENDER_ID].astype(str) + "→" + tmp[COL_RECEIVER_ID].astype(str)
    near = (
        tmp.groupby(["pair", "month_id"], observed=True)["sig_near_thr"]
        .mean()
        .reset_index(name="sig_near_thr")
        .assign(hit=lambda d: d["sig_near_thr"] > 0)
    )
    rep = (
        near.groupby("pair", as_index=False)["hit"]
        .sum()
        .rename(columns={"hit": "meses_con_near"})
        .query("meses_con_near >= @min_meses")
        .sort_values("meses_con_near", ascending=False)
    )
    return rep


def manager_conceptos_riesgo(reports, timeframe="todo_el_tiempo"):
    tx = _get_tx(reports, timeframe)[
        ["month_id", COL_RELATION, "nlp_concepto_sospechoso", "risk_score"]
    ].copy()
    if tx.empty:
        return tx
    mgr = tx[tx[COL_RELATION].str.contains("Manager", case=False, na=False)]
    top = (
        mgr.groupby(["month_id", "nlp_concepto_sospechoso"], as_index=False)
        .agg(
            tx_count=("risk_score", "count"),
            risk_p95=("risk_score", lambda s: float(s.quantile(0.95)) if len(s) else 0.0),
        )
        .sort_values(["month_id", "risk_p95", "tx_count"], ascending=[True, False, False])
    )
    return top


def prestamos_freq_dos_meses(reports, min_meses=2, timeframe="todo_el_tiempo"):
    tx = _get_tx(reports, timeframe)
    if tx.empty or "month_id" not in tx:
        return pd.DataFrame(columns=["pair", "meses_condicion"])
    tmp = tx[["month_id", COL_SENDER_ID, COL_RECEIVER_ID, "sig_loan_bad_repay", "sig_freq"]].copy()
    tmp["pair"] = tmp[COL_SENDER_ID].astype(str) + "→" + tmp[COL_RECEIVER_ID].astype(str)
    mes_flag = (
        tmp.groupby(["pair", "month_id"], observed=True)
        .agg(loan_bad=("sig_loan_bad_repay", "max"), freq=("sig_freq", "mean"))
        .reset_index()
        .assign(hit=lambda d: d["loan_bad"] & (d["freq"] > 0))
    )
    candidatos = (
        mes_flag.groupby("pair", as_index=False)["hit"]
        .sum()
        .rename(columns={"hit": "meses_condicion"})
        .query("meses_condicion >= @min_meses")
        .sort_values("meses_condicion", ascending=False)
    )
    return candidatos


def smurf_cronico(reports, min_meses=3, timeframe="todo_el_tiempo"):
    tx = _get_tx(reports, timeframe)
    if tx.empty or "month_id" not in tx:
        return pd.DataFrame(columns=["pair", "meses_smurf"])
    tmp = tx[["month_id", COL_SENDER_ID, COL_RECEIVER_ID, "sig_smurf"]].copy()
    tmp["pair"] = tmp[COL_SENDER_ID].astype(str) + "→" + tmp[COL_RECEIVER_ID].astype(str)
    agg = (
        tmp.groupby(["pair", "month_id"], observed=True)["sig_smurf"]
        .mean()
        .reset_index(name="pct_smurf")
        .assign(hit=lambda d: d["pct_smurf"] > 0)
        .groupby("pair", as_index=False)["hit"]
        .sum()
        .rename(columns={"hit": "meses_smurf"})
        .sort_values("meses_smurf", ascending=False)
    )
    return agg
