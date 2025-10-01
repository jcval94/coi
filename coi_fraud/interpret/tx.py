
def tx_interpretation(r):
    msgs = []
    if r.get("feat_zscore_monto", 0) > 2.0:
        avg = r.get("feat_avg_monto_emisor", 0) or 0
        msgs.append(f"Monto atípico (z={r['feat_zscore_monto']:.2f}, prom≈${avg:,.0f}).")
    if "Manager" in str(r.get("relacion","")): msgs.append("Relación jefe–subordinado.")
    if bool(r.get("sig_yoyo", False)): msgs.append("Yo-Yo en ≤8h.")
    if bool(r.get("sig_smurf", False)): msgs.append("Smurfing (varios montos pequeños).")
    if bool(r.get("sig_loan_bad_repay", False)): msgs.append("Préstamo con repago insuficiente.")
    if bool(r.get("sig_freq", False)): msgs.append("Alta frecuencia en 30 días.")
    if bool(r.get("sig_recurrent", False)): msgs.append("Recurrente mensual.")
    if bool(r.get("sig_roundsum", False)): msgs.append("Monto redondo.")
    if bool(r.get("sig_near_thr", False)): msgs.append("Cerca de umbral.")
    if bool(r.get("p1_in_cycle", False)) or bool(r.get("p2_in_cycle", False)): msgs.append("Ciclo de fondos.")
    elif bool(r.get("p1_in_triangle", False)) or bool(r.get("p2_in_triangle", False)): msgs.append("Triángulos en red.")
    if r.get("nlp_concepto_sospechoso"): msgs.append(f"NLP: {r['nlp_concepto_sospechoso']}.")
    if (r.get("feat_nlp_vaguedad",0)>0.7) or (r.get("feat_nlp_emocion",0)>0): msgs.append("Descripción vaga/emocional.")
    parts=[("z",float(max(r.get("feat_zscore_monto",0),0))),("hier",0.8 if "Manager" in str(r.get("relacion","")) else 0.0),("nlp",float(r.get("feat_nlp_risk_points",0)))]
    top = [k for k,_ in sorted(parts, key=lambda kv: kv[1], reverse=True)[:3]]
    return f"Principal: {' + '.join(top)}. " + " ".join(msgs[:4])
