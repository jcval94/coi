
def person_interpretation(r):
    net = float(r["sum_emit"] - r["sum_recv"])
    dir_txt = "emite más" if net>0 else ("recibe más" if net<0 else "equilibrado")
    tier = ("CRITICO" if r["risk_avg_person"]>=4.7 else "ALTO" if r["risk_avg_person"]>=3.2 else "MEDIO" if r["risk_avg_person"]>=1.8 else "BAJO")
    return (f"Emite {int(r['n_tx_emit'])} tx (${r['sum_emit']:,.0f}) y recibe {int(r['n_tx_recv'])} tx (${r['sum_recv']:,.0f}); "
            f"{dir_txt}. Riesgo promedio: {tier}.")
