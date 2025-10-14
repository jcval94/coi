
from ..config import P


def tx_interpretation(r):
    weights = P.weights
    msgs = []
    parts = []

    def _add_part(key: str, value: float):
        if value > 0:
            parts.append((key, float(value)))

    zscore = float(r.get("feat_zscore_monto", 0) or 0)
    if zscore > 2.0:
        avg = r.get("feat_avg_monto_emisor", 0) or 0
        msgs.append(
            f"Monto atípico (z={zscore:.2f}, prom≈${avg:,.0f})."
        )
        _add_part("z", max(zscore, 0.0) * weights.get("zscore", 0.0))

    relacion = str(r.get("relacion", ""))
    hier_flag = "Manager" in relacion
    if hier_flag:
        msgs.append("Relación jefe–subordinado.")
        _add_part("hier", weights.get("hierarchy", 0.0))

    if bool(r.get("sig_yoyo", False)):
        msgs.append("Yo-Yo en ≤8h.")
        _add_part("yoyo", weights.get("yoyo", 0.0))
    if bool(r.get("sig_smurf", False)):
        msgs.append("Smurfing (varios montos pequeños).")
        _add_part("smurf", weights.get("smurf", 0.0))
    if bool(r.get("sig_loan_bad_repay", False)):
        msgs.append("Préstamo con repago insuficiente.")
        _add_part("loan", weights.get("loan", 0.0))
    if bool(r.get("sig_freq", False)):
        msgs.append("Alta frecuencia en 30 días.")
        _add_part("freq", weights.get("freq", 0.0))
    if bool(r.get("sig_recurrent", False)):
        msgs.append("Recurrente mensual.")
        _add_part("recurrent", weights.get("recurrent", 0.0))
    if bool(r.get("sig_roundsum", False)):
        msgs.append("Monto redondo.")
        _add_part("round", weights.get("roundsum", 0.0))
    if bool(r.get("sig_near_thr", False)):
        msgs.append("Cerca de umbral.")
        _add_part("nearthr", weights.get("nearthr", 0.0))

    in_cycle = bool(r.get("p1_in_cycle", False)) or bool(r.get("p2_in_cycle", False))
    in_triangle = bool(r.get("p1_in_triangle", False)) or bool(
        r.get("p2_in_triangle", False)
    )
    if in_cycle:
        msgs.append("Ciclo de fondos.")
        _add_part("cycle", weights.get("sna_cycle", 0.0))
    elif in_triangle:
        msgs.append("Triángulos en red.")
        _add_part("triangle", weights.get("sna_triangle", 0.0))

    if bool(r.get("sig_quid_pro_quo", False)):
        score = float(r.get("feat_quid_score", 0) or 0)
        msgs.append(f"Posible algo por algo (score≈{score:.2f}).")
        _add_part("quid", weights.get("quid", 0.0))

    if bool(r.get("sig_reference_reuse", False)):
        msgs.append("Referencia reutilizada entre pares.")
        _add_part("ref", weights.get("reference_reuse", 0.0))

    if bool(r.get("sig_pair_change_point", False)):
        msgs.append("Cambio brusco en la serie del par.")
        _add_part("chg", weights.get("change_point", 0.0))

    if bool(r.get("sig_pair_new_edge", False)):
        gap = r.get("feat_pair_months_since_prev")
        if gap is not None and gap == gap:
            msgs.append(
                f"Nuevo enlace tras {gap:.0f} meses sin transacciones."
            )
        else:
            msgs.append("Nuevo enlace tras inactividad prolongada.")
        _add_part("new", weights.get("new_edge", 0.0))

    vaguedad = float(r.get("feat_nlp_vaguedad", 0) or 0)
    emocion = float(r.get("feat_nlp_emocion", 0) or 0)
    sentimiento = float(r.get("feat_nlp_sentimiento", 0) or 0)
    coi_score = float(r.get("feat_nlp_coi_score", 0) or 0)
    risk_points = float(r.get("feat_nlp_risk_points", 0) or 0)
    evento_corp = bool(r.get("nlp_evento_corporativo", False))
    nlp_detail_added = False
    if r.get("nlp_concepto_sospechoso"):
        msgs.append(f"NLP: {r['nlp_concepto_sospechoso']}.")
        nlp_detail_added = True
    if vaguedad > 0.7 or emocion > 0:
        msgs.append("Descripción vaga/emocional.")
        nlp_detail_added = True
    if abs(sentimiento) >= 0.3:
        tono = "positivo" if sentimiento > 0 else "negativo"
        msgs.append(f"Sentimiento {tono} inusual (score={sentimiento:.2f}).")
        nlp_detail_added = True
    if evento_corp:
        msgs.append("Menciona evento corporativo sensible.")
        nlp_detail_added = True
    if coi_score >= 2.0:
        msgs.append(f"Score COI elevado ({coi_score:.2f}).")
        nlp_detail_added = True
    nlp_part = (
        (vaguedad + (0.5 if emocion > 0 else 0.0)) * 0.45 * weights.get("nlp", 0.0)
        + (risk_points * 0.12)
        + (abs(sentimiento) * 0.25)
        + (coi_score * 0.18)
        + (0.2 if evento_corp else 0.0)
    )
    if nlp_part > 0:
        _add_part("nlp", nlp_part)
        if not nlp_detail_added:
            msgs.append("Indicadores NLP elevados.")

    parts.sort(key=lambda kv: kv[1], reverse=True)
    label_map = {
        "z": "monto atípico",
        "hier": "jerarquía",
        "nlp": "texto riesgoso",
        "yoyo": "yo-yo",
        "smurf": "smurfing",
        "loan": "préstamo irregular",
        "freq": "alta frecuencia",
        "recurrent": "recurrente",
        "round": "monto redondo",
        "nearthr": "cerca de umbral",
        "cycle": "ciclo de fondos",
        "triangle": "triángulos",
        "quid": "algo por algo",
        "ref": "referencia reutilizada",
        "chg": "cambio brusco",
        "new": "nuevo enlace",
    }
    top = [label_map[k] for k, _ in parts[:3] if k in label_map]

    principal = (
        f"Principal: {' + '.join(top)}. "
        if top
        else "Principal: sin señales destacadas. "
    )
    detalle = " ".join(msgs[:4]) if msgs else "Sin señales descriptivas."
    return principal + detalle
