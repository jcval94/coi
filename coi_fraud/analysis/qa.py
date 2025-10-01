
import re, pandas as pd
def empleados_recepcion_constante(reports, min_meses=3):
    d = reports["agg_par_mensual"][["month_id","pair","tx_count","pct_recurrent"]].copy()
    out = (d.assign(tiene_tx=d["tx_count"]>0, tiene_rec=d["pct_recurrent"]>0)
             .groupby("pair", as_index=False)
             .agg(meses_con_tx=("tiene_tx","sum"), meses_recurrente=("tiene_rec","sum"))
             .query("meses_con_tx >= @min_meses")
             .sort_values(["meses_recurrente","meses_con_tx"], ascending=[False,False]))
    return out

def desbalance_personas(reports):
    pers = reports["agg_persona_mensual"].copy()
    desbalance = (pers.assign(net=pers["sum_emit"]-pers["sum_recv"], ratio=(pers["sum_emit"]/(pers["sum_recv"]+1e-9)))
                     .sort_values(["month_id","net"], ascending=[True, False]))
    return desbalance

def manager_conceptos_sospechosos(reports, keywords=None):
    tx = reports["tx_transacciones_priorizadas"].copy()
    mgr = tx[tx["relacion"].str.contains("Manager", case=False, na=False)]
    if keywords:
        pat = re.compile("|".join([re.escape(k) for k in keywords]), re.IGNORECASE)
        mgr = mgr[mgr["descripcion"].fillna("").str.contains(pat)]
    agg = (mgr.groupby(["month_id","nlp_concepto_sospechoso"], as_index=False)
           .agg(tx_count=("risk_score","count"),
                risk_p95=("risk_score", lambda s: float(s.quantile(0.95)) if len(s) else 0.0))
           .sort_values(["month_id","risk_p95","tx_count"], ascending=[True,False,False]))
    return agg

def centralizadores(reports):
    tx = reports["tx_transacciones_priorizadas"][["month_id","persona_1","persona_2","monto","risk_score"]].copy()
    centro = (tx.groupby(["month_id","persona_2"], as_index=False)
        .agg(inflow=("monto","sum"), emisores_unicos=("persona_1","nunique"), n_tx=("monto","count"), risk_avg=("risk_score","mean"))
        .assign(centralidad=lambda d: d["inflow"]*d["emisores_unicos"])
        .sort_values(["month_id","centralidad"], ascending=[True,False]))
    return centro

def yoyo_consecutivos(reports):
    par = reports["agg_par_mensual"][["month_id","pair","pct_yoyo"]].copy()
    par["hit"] = par["pct_yoyo"] > 0
    out = (par.sort_values(["pair","month_id"]).groupby("pair")["hit"].apply(lambda s: any(s.shift(1).fillna(False) & s)).reset_index(name="tiene_racha_yo_yo"))
    return out[out["tiene_racha_yo_yo"]]

def near_thr_repetido(reports, min_meses=2):
    tx = reports["tx_transacciones_priorizadas"][["month_id","persona_1","persona_2","sig_near_thr"]].copy()
    tx["pair"] = tx["persona_1"] + "→" + tx["persona_2"]
    near = tx.groupby(["pair","month_id"], as_index=False)["sig_near_thr"].mean().assign(hit=lambda d: d["sig_near_thr"]>0)
    rep = near.groupby("pair", as_index=False)["hit"].sum().rename(columns={"hit":"meses_con_near"}).query("meses_con_near >= @min_meses").sort_values("meses_con_near", ascending=False)
    return rep

def manager_conceptos_riesgo(reports):
    tx = reports["tx_transacciones_priorizadas"][["month_id","relacion","nlp_concepto_sospechoso","risk_score"]].copy()
    mgr = tx[tx["relacion"].str.contains("Manager", case=False, na=False)]
    top = (mgr.groupby(["month_id","nlp_concepto_sospechoso"], as_index=False)
         .agg(tx_count=("risk_score","count"), risk_p95=("risk_score", lambda s: float(s.quantile(0.95)) if len(s) else 0.0))
         .sort_values(["month_id","risk_p95","tx_count"], ascending=[True, False, False]))
    return top

def prestamos_freq_dos_meses(reports, min_meses=2):
    tx = reports["tx_transacciones_priorizadas"][["month_id","persona_1","persona_2","sig_loan_bad_repay","sig_freq"]].copy()
    tx["pair"] = tx["persona_1"] + "→" + tx["persona_2"]
    mes_flag = (tx.groupby(["pair","month_id"], as_index=False)
                  .agg(loan_bad=("sig_loan_bad_repay","max"), freq=("sig_freq","mean"))
                  .assign(hit=lambda d: d["loan_bad"] & (d["freq"]>0)))
    candidatos = mes_flag.groupby("pair", as_index=False)["hit"].sum().rename(columns={"hit":"meses_condicion"}).query("meses_condicion >= @min_meses").sort_values("meses_condicion", ascending=False)
    return candidatos

def smurf_cronico(reports, min_meses=3):
    par = reports["agg_par_mensual"][["pair","month_id","pct_smurf"]].copy()
    cron = (par.assign(hit=par["pct_smurf"]>0).groupby("pair", as_index=False)["hit"].sum().rename(columns={"hit":"meses_smurf"}).query("meses_smurf >= @min_meses").sort_values("meses_smurf", ascending=False))
    return cron
