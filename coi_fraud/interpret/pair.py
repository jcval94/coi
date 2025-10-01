
def pair_interpretation(r):
    sigs=[]
    if r["pct_yoyo"]>0: sigs.append("yo-yo")
    if r["pct_smurf"]>0: sigs.append("smurfing")
    if r["pct_freq"]>0: sigs.append("alta frecuencia")
    if r.get("pct_recurrent",0)>0: sigs.append("recurrente")
    if r.get("pct_quid_pro_quo",0)>0: sigs.append("quid-pro-quo")
    if r.get("pct_reference_reuse",0)>0: sigs.append("referencia repetida")
    sigs_txt = ", ".join(sigs) if sigs else "sin señales fuertes"
    tier = ("CRITICO" if r["risk_max"]>=4.7 else "ALTO" if r["risk_max"]>=3.2 else "MEDIO" if r["risk_max"]>=1.8 else "BAJO")
    return (f"{int(r['tx_count'])} tx por ${r['tx_sum']:,.0f}. Señales: {sigs_txt}. Riesgo máx: {tier}.")
