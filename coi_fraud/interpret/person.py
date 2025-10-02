
def person_interpretation(r):
    net = float(r["sum_emit"] - r["sum_recv"])
    dir_txt = "emite más" if net > 0 else ("recibe más" if net < 0 else "equilibrado")
    tier = (
        "CRITICO"
        if r["risk_avg_person"] >= 4.7
        else "ALTO"
        if r["risk_avg_person"] >= 3.2
        else "MEDIO"
        if r["risk_avg_person"] >= 1.8
        else "BAJO"
    )
    extras = []
    coi_score = float(r.get("nlp_persona_score_prob_coi", 0) or 0)
    if coi_score >= 0.5:
        extras.append(f"score COI {coi_score:.2f}")
    senti_avg = float(r.get("nlp_persona_sentimiento_promedio", 0) or 0)
    if abs(senti_avg) >= 0.1:
        tono = "positivo" if senti_avg > 0 else "negativo"
        extras.append(f"sentimiento {tono} {senti_avg:.2f}")
    extras_txt = f" Indicadores NLP: {', '.join(extras)}." if extras else ""
    return (
        f"Emite {int(r['n_tx_emit'])} tx (${r['sum_emit']:,.0f}) y recibe {int(r['n_tx_recv'])} tx (${r['sum_recv']:,.0f}); "
        f"{dir_txt}. Riesgo promedio: {tier}.{extras_txt}"
    )
