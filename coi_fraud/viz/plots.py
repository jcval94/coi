import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_RELATION, COL_SENDER_ID


def plot_constant_receipt_heatmap(reports, ax=None):
    df = reports["agg_par_mensual"][
        ["month_id", "pair", "tx_count"]
    ].copy()
    pivot = df.pivot_table(
        index="pair",
        columns="month_id",
        values="tx_count",
        fill_value=0,
        aggfunc="sum",
    )
    ax = ax or plt.figure().gca()
    sns.heatmap(pivot, ax=ax)
    ax.set_title("Constancia de recepción por par (tx_count por mes)")
    ax.set_xlabel("month_id")
    ax.set_ylabel("pair")
    return ax


def plot_person_imbalance_bar(reports, month_id=None, ax=None):
    df = reports["agg_persona_mensual"].copy()
    if month_id is None and len(df):
        month_id = df["month_id"].iloc[0]
    dd = df[df["month_id"] == month_id].copy()
    dd["net"] = dd["sum_emit"] - dd["sum_recv"]
    dd = dd.sort_values("net", ascending=False).head(25)
    ax = ax or plt.figure().gca()
    sns.barplot(data=dd, x="net", y="persona", ax=ax)
    ax.set_title(f"Desbalance emite-recibe (top 25) — {month_id}")
    ax.set_xlabel("neto")
    ax.set_ylabel("persona")
    return ax


def plot_manager_concepts_bar(reports, month_id=None, ax=None):
    tx = reports["tx_transacciones_priorizadas"].copy()
    mgr = tx[tx[COL_RELATION].str.contains("Manager", case=False, na=False)]
    if month_id is not None:
        mgr = mgr[mgr["month_id"] == month_id]
    agg = (
        mgr.groupby("nlp_concepto_sospechoso")[COL_AMOUNT]
        .count()
        .sort_values(ascending=False)
        .reset_index(name="tx_count")
    )
    ax = ax or plt.figure().gca()
    sns.barplot(data=agg, y="nlp_concepto_sospechoso", x="tx_count", ax=ax)
    ax.set_title("Relaciones Manager — conceptos NLP")
    ax.set_xlabel("tx_count")
    ax.set_ylabel("concepto")
    return ax


def plot_centralizer_scatter(reports, month_id=None, ax=None):
    tx = reports["tx_transacciones_priorizadas"][
        ["month_id", COL_SENDER_ID, COL_RECEIVER_ID, COL_AMOUNT, "risk_score"]
    ].copy()
    dd = (
        tx.groupby(["month_id", COL_RECEIVER_ID], as_index=False)
        .agg(
            inflow=(COL_AMOUNT, "sum"),
            emisores_unicos=(COL_SENDER_ID, "nunique"),
            n_tx=(COL_AMOUNT, "count"),
            risk_avg=("risk_score", "mean"),
        )
    )
    if month_id is None and len(dd):
        month_id = dd["month_id"].iloc[0]
    dd = dd[dd["month_id"] == month_id]
    ax = ax or plt.figure().gca()
    sns.scatterplot(
        data=dd,
        x="emisores_unicos",
        y="inflow",
        size="n_tx",
        hue="risk_avg",
        ax=ax,
    )
    ax.set_title(f"Centralización de inflow — {month_id}")
    ax.set_xlabel("emisores_unicos")
    ax.set_ylabel("inflow")
    return ax


def plot_yoyo_pairs_timeline(reports, ax=None):
    par = reports["agg_par_mensual"][
        ["month_id", "pair", "pct_yoyo"]
    ].copy()
    top_pairs = (
        par.groupby("pair")["pct_yoyo"].mean().sort_values(ascending=False).head(10).index.tolist()
    )
    dd = par[par["pair"].isin(top_pairs)]
    ax = ax or plt.figure().gca()
    sns.lineplot(data=dd, x="month_id", y="pct_yoyo", hue="pair", marker="o", ax=ax)
    ax.set_title("Yo-Yo por mes (top 10 pares)")
    ax.set_xlabel("month_id")
    ax.set_ylabel("pct_yoyo")
    return ax


def plot_near_threshold_heatmap(reports, ax=None):
    tx = reports["tx_transacciones_priorizadas"][
        ["month_id", COL_SENDER_ID, COL_RECEIVER_ID, "sig_near_thr"]
    ].copy()
    tx["pair"] = tx[COL_SENDER_ID] + "→" + tx[COL_RECEIVER_ID]
    near = tx.groupby(["pair", "month_id"], as_index=False)["sig_near_thr"].mean()
    pivot = near.pivot_table(
        index="pair",
        columns="month_id",
        values="sig_near_thr",
        fill_value=0,
        aggfunc="mean",
    )
    ax = ax or plt.figure().gca()
    sns.heatmap(pivot, ax=ax)
    ax.set_title("Cerca de umbral — intensidad por par/mes")
    ax.set_xlabel("month_id")
    ax.set_ylabel("pair")
    return ax


def plot_manager_concepts_risk(reports, ax=None):
    tx = reports["tx_transacciones_priorizadas"][
        ["month_id", COL_RELATION, "nlp_concepto_sospechoso", "risk_score"]
    ].copy()
    mgr = tx[tx[COL_RELATION].str.contains("Manager", case=False, na=False)]
    agg = (
        mgr.groupby(["month_id", "nlp_concepto_sospechoso"], as_index=False)
        .agg(
            tx_count=("risk_score", "count"),
            risk_p95=("risk_score", lambda s: float(s.quantile(0.95)) if len(s) else 0.0),
        )
    )
    ax = ax or plt.figure().gca()
    sns.scatterplot(
        data=agg,
        x="risk_p95",
        y="tx_count",
        hue="nlp_concepto_sospechoso",
        style="month_id",
        ax=ax,
    )
    ax.set_title("Manager: conceptos vs severidad (p95)")
    ax.set_xlabel("risk_p95")
    ax.set_ylabel("tx_count")
    return ax


def plot_loan_freq_dual(reports, ax=None):
    tx = reports["tx_transacciones_priorizadas"][
        ["month_id", COL_SENDER_ID, COL_RECEIVER_ID, "sig_loan_bad_repay", "sig_freq"]
    ].copy()
    tx["pair"] = tx[COL_SENDER_ID] + "→" + tx[COL_RECEIVER_ID]
    mes_flag = (
        tx.groupby(["pair", "month_id"], as_index=False)
        .agg(loan_bad=("sig_loan_bad_repay", "max"), freq=("sig_freq", "mean"))
        .assign(hit=lambda d: d["loan_bad"] & (d["freq"] > 0))
    )
    agg = (
        mes_flag.groupby("pair", as_index=False)["hit"]
        .sum()
        .rename(columns={"hit": "meses_condicion"})
        .sort_values("meses_condicion", ascending=False)
        .head(25)
    )
    ax = ax or plt.figure().gca()
    sns.barplot(data=agg, x="meses_condicion", y="pair", ax=ax)
    ax.set_title("Préstamo sin repago + alta frecuencia")
    ax.set_xlabel("meses_condicion")
    ax.set_ylabel("pair")
    return ax


def plot_smurf_chronic(reports, ax=None):
    par = reports["agg_par_mensual"][
        ["pair", "month_id", "pct_smurf"]
    ].copy()
    agg = (
        par.assign(hit=par["pct_smurf"] > 0)
        .groupby("pair", as_index=False)["hit"]
        .sum()
        .rename(columns={"hit": "meses_smurf"})
        .sort_values("meses_smurf", ascending=False)
        .head(25)
    )
    ax = ax or plt.figure().gca()
    sns.barplot(data=agg, x="meses_smurf", y="pair", ax=ax)
    ax.set_title("Smurfing crónico (meses con señal)")
    ax.set_xlabel("meses_smurf")
    ax.set_ylabel("pair")
    return ax
