import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..schemas import COL_AMOUNT, COL_RECEIVER_ID, COL_RELATION, COL_SENDER_ID


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


def _ensure_ax(ax):
    return ax if ax is not None else plt.figure().gca()


def plot_constant_receipt_heatmap(reports, timeframe="todo_el_tiempo", ax=None):
    tx = _get_tx(reports, timeframe)
    ax = _ensure_ax(ax)
    if tx.empty or "month_id" not in tx:
        ax.set_title(f"Constancia de recepción — {timeframe} (sin datos)")
        ax.set_xlabel("month_id")
        ax.set_ylabel("pair")
        return ax
    tmp = tx.copy()
    tmp["pair"] = tmp[COL_SENDER_ID].astype(str) + "→" + tmp[COL_RECEIVER_ID].astype(str)
    monthly = (
        tmp.groupby(["month_id", "pair"], observed=True)[COL_AMOUNT]
        .count()
        .reset_index(name="tx_count")
    )
    pivot = monthly.pivot_table(
        index="pair",
        columns="month_id",
        values="tx_count",
        fill_value=0,
        aggfunc="sum",
    )
    sns.heatmap(pivot, ax=ax)
    ax.set_title(f"Constancia de recepción por par — {timeframe}")
    ax.set_xlabel("month_id")
    ax.set_ylabel("pair")
    return ax


def plot_person_imbalance_bar(reports, timeframe="todo_el_tiempo", ax=None, top_n=25):
    df = _get_section(reports, "persona", timeframe)
    ax = _ensure_ax(ax)
    if df.empty:
        ax.set_title(f"Desbalance emite-recibe — {timeframe} (sin datos)")
        return ax
    dd = df.copy()
    dd["net"] = dd["sum_emit"] - dd["sum_recv"]
    dd = dd.sort_values("net", ascending=False).head(top_n)
    sns.barplot(data=dd, x="net", y="persona", ax=ax)
    ax.set_title(f"Desbalance emite-recibe (top {top_n}) — {timeframe}")
    ax.set_xlabel("neto")
    ax.set_ylabel("persona")
    return ax


def plot_manager_concepts_bar(reports, timeframe="todo_el_tiempo", ax=None):
    tx = _get_tx(reports, timeframe)
    ax = _ensure_ax(ax)
    if tx.empty:
        ax.set_title(f"Relaciones Manager — conceptos NLP ({timeframe}) sin datos")
        return ax
    mgr = tx[tx[COL_RELATION].str.contains("Manager", case=False, na=False)].copy()
    agg = (
        mgr.groupby("nlp_concepto_sospechoso")[COL_AMOUNT]
        .count()
        .sort_values(ascending=False)
        .reset_index(name="tx_count")
    )
    sns.barplot(data=agg, y="nlp_concepto_sospechoso", x="tx_count", ax=ax)
    ax.set_title(f"Relaciones Manager — conceptos NLP ({timeframe})")
    ax.set_xlabel("tx_count")
    ax.set_ylabel("concepto")
    return ax


def plot_centralizer_scatter(reports, timeframe="todo_el_tiempo", ax=None):
    tx = _get_tx(reports, timeframe)[
        ["month_id", COL_SENDER_ID, COL_RECEIVER_ID, COL_AMOUNT, "risk_score"]
    ].copy()
    ax = _ensure_ax(ax)
    if tx.empty:
        ax.set_title(f"Centralización de inflow — {timeframe} (sin datos)")
        return ax
    dd = (
        tx.groupby(["month_id", COL_RECEIVER_ID], as_index=False)
        .agg(
            inflow=(COL_AMOUNT, "sum"),
            emisores_unicos=(COL_SENDER_ID, "nunique"),
            n_tx=(COL_AMOUNT, "count"),
            risk_avg=("risk_score", "mean"),
        )
    )
    sns.scatterplot(
        data=dd,
        x="emisores_unicos",
        y="inflow",
        size="n_tx",
        hue="risk_avg",
        ax=ax,
    )
    ax.set_title(f"Centralización de inflow — {timeframe}")
    ax.set_xlabel("emisores_unicos")
    ax.set_ylabel("inflow")
    return ax


def plot_yoyo_pairs_timeline(reports, timeframe="todo_el_tiempo", ax=None, top_n=10):
    tx = _get_tx(reports, timeframe)
    ax = _ensure_ax(ax)
    if tx.empty or "month_id" not in tx:
        ax.set_title(f"Yo-Yo por mes — {timeframe} (sin datos)")
        return ax
    tmp = tx.copy()
    tmp["pair"] = tmp[COL_SENDER_ID].astype(str) + "→" + tmp[COL_RECEIVER_ID].astype(str)
    monthly = (
        tmp.groupby(["month_id", "pair"], observed=True)["sig_yoyo"]
        .mean()
        .reset_index(name="pct_yoyo")
    )
    top_pairs = (
        monthly.groupby("pair")["pct_yoyo"].mean().sort_values(ascending=False).head(top_n).index.tolist()
    )
    dd = monthly[monthly["pair"].isin(top_pairs)]
    sns.lineplot(data=dd, x="month_id", y="pct_yoyo", hue="pair", marker="o", ax=ax)
    ax.set_title(f"Yo-Yo por mes (top {top_n} pares) — {timeframe}")
    ax.set_xlabel("month_id")
    ax.set_ylabel("pct_yoyo")
    return ax


def plot_near_threshold_heatmap(reports, timeframe="todo_el_tiempo", ax=None):
    tx = _get_tx(reports, timeframe)
    ax = _ensure_ax(ax)
    if tx.empty or "month_id" not in tx:
        ax.set_title(f"Cerca de umbral — {timeframe} (sin datos)")
        ax.set_xlabel("month_id")
        ax.set_ylabel("pair")
        return ax
    tmp = tx[["month_id", COL_SENDER_ID, COL_RECEIVER_ID, "sig_near_thr"]].copy()
    tmp["pair"] = tmp[COL_SENDER_ID].astype(str) + "→" + tmp[COL_RECEIVER_ID].astype(str)
    near = tmp.groupby(["pair", "month_id"], observed=True)["sig_near_thr"].mean().reset_index()
    pivot = near.pivot_table(
        index="pair",
        columns="month_id",
        values="sig_near_thr",
        fill_value=0,
        aggfunc="mean",
    )
    sns.heatmap(pivot, ax=ax)
    ax.set_title(f"Cerca de umbral — intensidad por par/mes ({timeframe})")
    ax.set_xlabel("month_id")
    ax.set_ylabel("pair")
    return ax


def plot_manager_concepts_risk(reports, timeframe="todo_el_tiempo", ax=None):
    tx = _get_tx(reports, timeframe)[
        ["month_id", COL_RELATION, "nlp_concepto_sospechoso", "risk_score"]
    ].copy()
    ax = _ensure_ax(ax)
    if tx.empty:
        ax.set_title(f"Manager: conceptos vs severidad ({timeframe}) sin datos")
        return ax
    mgr = tx[tx[COL_RELATION].str.contains("Manager", case=False, na=False)]
    agg = (
        mgr.groupby(["month_id", "nlp_concepto_sospechoso"], as_index=False)
        .agg(
            tx_count=("risk_score", "count"),
            risk_p95=("risk_score", lambda s: float(s.quantile(0.95)) if len(s) else 0.0),
        )
    )
    sns.scatterplot(
        data=agg,
        x="risk_p95",
        y="tx_count",
        hue="nlp_concepto_sospechoso",
        style="month_id",
        ax=ax,
    )
    ax.set_title(f"Manager: conceptos vs severidad (p95) — {timeframe}")
    ax.set_xlabel("risk_p95")
    ax.set_ylabel("tx_count")
    return ax


def plot_loan_freq_dual(reports, timeframe="todo_el_tiempo", ax=None, top_n=25):
    tx = _get_tx(reports, timeframe)
    ax = _ensure_ax(ax)
    if tx.empty or "month_id" not in tx:
        ax.set_title(f"Préstamo sin repago + alta frecuencia — {timeframe} (sin datos)")
        return ax
    tmp = tx[["month_id", COL_SENDER_ID, COL_RECEIVER_ID, "sig_loan_bad_repay", "sig_freq"]].copy()
    tmp["pair"] = tmp[COL_SENDER_ID].astype(str) + "→" + tmp[COL_RECEIVER_ID].astype(str)
    mes_flag = (
        tmp.groupby(["pair", "month_id"], observed=True)
        .agg(loan_bad=("sig_loan_bad_repay", "max"), freq=("sig_freq", "mean"))
        .reset_index()
        .assign(hit=lambda d: d["loan_bad"] & (d["freq"] > 0))
    )
    agg = (
        mes_flag.groupby("pair", as_index=False)["hit"]
        .sum()
        .rename(columns={"hit": "meses_condicion"})
        .sort_values("meses_condicion", ascending=False)
        .head(top_n)
    )
    sns.barplot(data=agg, x="meses_condicion", y="pair", ax=ax)
    ax.set_title(f"Préstamo sin repago + alta frecuencia — {timeframe}")
    ax.set_xlabel("meses_condicion")
    ax.set_ylabel("pair")
    return ax


def plot_smurf_chronic(reports, timeframe="todo_el_tiempo", ax=None, top_n=25):
    tx = _get_tx(reports, timeframe)
    ax = _ensure_ax(ax)
    if tx.empty or "month_id" not in tx:
        ax.set_title(f"Smurfing crónico — {timeframe} (sin datos)")
        return ax
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
        .head(top_n)
    )
    sns.barplot(data=agg, x="meses_smurf", y="pair", ax=ax)
    ax.set_title(f"Smurfing crónico (meses con señal) — {timeframe}")
    ax.set_xlabel("meses_smurf")
    ax.set_ylabel("pair")
    return ax
