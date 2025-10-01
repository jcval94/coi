
# (Función completa incluida; ver categoría FAVORES_SEXUALES agregada)
from typing import Optional
def nlp_mx_etiquetar_transacciones_pro(
    df,
    col_texto,
    col_relacion: Optional[str]=None,
    use_embeddings=True,
    return_similitudes_top=5,
):
    import re, math, unicodedata, pandas as pd
    from difflib import SequenceMatcher

    if col_texto not in df.columns:
        raise ValueError(f"La columna '{col_texto}' no existe en el DataFrame.")
    has_rel = col_relacion in df.columns if col_relacion else False

    ZWSP_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF]")
    EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]")
    ACCENT_MAP = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
    LEET_MAP = {"0":"o","1":"i","3":"e","4":"a","5":"s","7":"t","$":"s","@":"a","!":"i"}

    def norm_basic(s: str) -> str:
        s = "" if s is None else str(s)
        s = ZWSP_PATTERN.sub("", s)
        s = unicodedata.normalize("NFKC", s)
        s = s.translate(ACCENT_MAP).lower().strip()
        s = "".join(LEET_MAP.get(ch, ch) for ch in s)
        s = re.sub(r"\s+", " ", s)
        return s

    def tokens(s: str):
        return re.findall(r"[a-z0-9]+", s)

    def has_emoji(raw: str) -> bool:
        return bool(EMOJI_PATTERN.search(raw or ""))

    LEX = {
        "SOBORNO": {"peso": 3.4, "palabras": ["mordida","moche","mochada","coima","aceitar","engrasar","chayote",
            "agradecimiento","detalle","regalito","incentivo","comision","comisión","bono especial","por tu gestion","por tu gestión",
            "por el paro","por sacar la chamba","palomear","para el chesco","para las cocas","para el cafe","para la gasolina","por fuera","dar aceite"],
            "patrones": [r"\\b(agradecimiento|detalle|incentivo|comisi[oó]n|bono|mordida|moche|coima|engras|aceit|chayote)\\b.*\\b(aprob|firm|palome|autoriza|asigna|libera)\\w*",
                         r"\\b(por|para)\\b.*\\b(aprob|firm|palome|autoriza|asigna|libera)\\w*"],
            "seeds": ["agradecimiento por aprobar la orden de compra","incentivo por firmar contrato","detalle para que se libere la orden","mordida para autorizar proveedor","moche por cerrar el trato","por fuera para agilizar la firma"]},
        "EXTORSION": {"peso": 3.1, "palabras": ["cuota","derecho de piso","aportacion","aportación","cooperacion","cooperación","alinearse","afloja","no te cierres","no me quemes","para que no haya problema","no te atraso","no te atrase"],
            "patrones":[r"\\b(cuota|aportaci[oó]n|cooperaci[oó]n|derecho de piso)\\b.*\\b(evaluaci[oó]n|proyecto|asignaci[oó]n)\\b",
                        r"(para que no haya problema|no te atraso|no te atrase)"],
            "seeds":["cuota mensual para que no haya problema con tu evaluación","aportación para no atrasar el proyecto","derecho de piso para evitar problemas"]},
        "OFUSCACION":{"peso":2.9,"palabras":["sin cfdi","sin factura","no timbrar","no timbres","por fuera","en sobre","bajo el agua","discretito","discreto","por la sombrita","off the record"],
            "patrones":[r"(sin\\s+cfdi|sin\\s+factura|no\\s+timbrar|no\\s+timbres|por\\s+fuera|en\\s+sobre|bajo\\s+el\\s+agua|off\\s+the\\s+record)"],
            "seeds":["pago sin cfdi por fuera","en sobre y sin factura","off the record sin timbrar"]},
        "FACILITACION":{"peso":2.3,"palabras":["agilizar","destrabar","desatorar","prioridad","darle salida","liberar","gestion","gestión","tramite","trámite","fast track","greenlight","palomear"],
            "patrones":[], "eventos":["oc","po","orden de compra","licitacion","licitación","alta proveedor","proveedor","contrato","firma"],
            "seeds":["pago para agilizar la orden de compra","gestión para liberar alta de proveedor","fast track del contrato"]},
        "NOMINA_PARALELA":{"peso":1.9,"palabras":["bono","comision","comisión","incentivo","premio","gratificacion","gratificación"],"patrones":[], "seeds":["bono especial fuera de nómina","comisión por el contrato pagada por fuera"]},
        "REEMBOLSO_DUDOSO":{"peso":1.7,"palabras":["reembolso","reembolsos","viaticos","viáticos","gastos","varios","servicio","servicios","material","caja chica"],
            "patrones":[], "seeds":["reembolso gastos sin detalle","viáticos varios","gastos servicio"]},
        "PRESTAMO":{"peso":1.8,"palabras":["prestamo","préstamo","adelanto","abono","saldo","liquidar","paguitos","en partes","depositito","pendiente mensual","lo del"],
            "patrones":[r"\\blo del\\s+\\d{1,2}\\b"],"seeds":["préstamo personal en partes","abono del 25","pendiente mensual del préstamo"]},
        "COI_RELACIONAL":{"peso":1.8,"palabras":["compadre","comadre","primo","sobrino","carnal","cuate","amigazo","de confianza","palanca","conecte","el de siempre","ya sabes quien","ya sabes quién"],
            "patrones":[], "seeds":["apoyo para compadre de compras","conecte de proveedor"]},
        "DINERO_SLANG":{"peso":1.0,"palabras":["lana","varo","feria","billete","pasta","morlacos","mangos","verdes","2k","5k","kilo"],"patrones":[r"\\b\\d+\\s*k\\b"],"seeds":["2k para el trámite","cinco mil para el cafe"]},
        "CODIGO":{"peso":1.2,"palabras":["c-azul","c-verde","c-naranja","px-","off the record"],"patrones":[r"\\b[a-z]{1,3}-\\d{1,4}\\b"],"seeds":["px-9 listo","c-azul ok"]},
        "DEIXIS":{"peso":1.1,"palabras":["lo de ayer","lo de antier","la vez pasada","como quedamos","igual que ayer","igual que antes","ya sabes","aquello","eso"],"patrones":[], "seeds":["lo de ayer igual","como quedamos"]},
        "EMOCIONAL":{"peso":0.9,"palabras":["gracias mil","favorzote","te debo la vida","eres el mejor","millones de gracias","increible","increíble","salvaste","confio en ti","confío en ti"],"patrones":[], "seeds":["gracias mil por el paro","favorzote con lo de la firma"]},
        "FAVORES_SEXUALES":{"peso":3.6,"palabras":[
            "favor sexual","sexo","salida intima","salida íntima","cariño especial","cita privada","encuentro","acompaname","acompáname","acompanamiento",
            "trato especial","detallito especial","algo mas","algo más"
            ],
            "patrones":[
                r"(favor|salida|cita|encuentro).{0,10}(intim|privad|personal)",
                r"(carin(?:o|\\xF3)|cari\\xF1o|mimos?).{0,10}(extra|especial)",
                r"(a\\s+cambio\\s+de).{0,12}(salida|cita|favor)",
                r"(trato\\s+especial).{0,20}(firma|aproba|libera|contrato|orden)"
            ],
            "seeds":["favor sexual por firma","salida íntima a cambio de aprobar","cita privada para liberar la orden","trato especial por el contrato"]
        },
    }

    EVENTOS_CLAVE = set(["oc","po","orden de compra","licitacion","licitación","alta proveedor","proveedor","auditoria","auditoría","evaluacion","evaluación","bono","calibracion","calibración","cierre","contrato","firma"])
    SOCIAL_WHITELIST = {"cumple","despedida","baby shower","boda","vaquita","coperacha","pastel","regalo despedida"}

    vectorizer = None; cat_centroids = {}
    if use_embeddings:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            docs = []; y = []
            for fam, spec in LEX.items():
                for s in spec.get("seeds", []): docs.append(norm_basic(s)); y.append(fam)
                for w in spec.get("palabras", []): docs.append(norm_basic(w)); y.append(fam)
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=1)
            X = vectorizer.fit_transform(docs)
            fam_to = {}
            for fam, row in zip(y, X):
                fam_to.setdefault(fam, []).append(row)
            for fam, mats in fam_to.items():
                cat_centroids[fam] = (mats[0] if len(mats)==1 else sum(mats)/len(mats))
        except Exception:
            vectorizer = None; cat_centroids = {}

    def vaguedad(tokens_list):
        VAG_TERMS = {"apoyo","gasto","gastos","reembolso","varios","servicio","servicios","material","proyecto","viaticos","viatico","viáticos","bono","comision","comisión"}
        if not tokens_list: return 1.0
        v = sum(1 for t in tokens_list if t in VAG_TERMS)/max(1,len(tokens_list))
        if len(tokens_list) <= 2: v = max(v, 0.8)
        return float(min(1.0, max(0.0, v)))

    def fuzzy_contains(haystack: str, needle: str, thresh=0.86) -> bool:
        if " " in needle and needle in haystack: return True
        if len(needle) < 4: return needle in haystack
        for m in re.finditer(r"[a-z0-9][a-z0-9\\s]{0,%d}" % max(3, len(needle)+6), haystack):
            seg = m.group(0)
            if SequenceMatcher(None, seg[:len(needle)+3], needle).ratio() >= thresh:
                return True
        return SequenceMatcher(None, haystack, needle).ratio() >= (thresh + 0.03)

    FAC_EVENT = re.compile(r"(agiliz|destrab|desator|prioridad|darle salida|liber|gesti[oó]n|tramite|tr[aá]mite|fast[- ]?track|greenlight|palomear)")
    EVENT_WORDS = re.compile(r"(?:\\boc\\b|\\bpo\\b|orden de compra|licitaci[oó]n|alta proveedor|proveedor|contrato|firma)")

    def match_families(norm_text, toks):
        categorias = []; frases = []; contrib = {}
        evento_presente = bool(EVENT_WORDS.search(norm_text))
        social_hit = any(w in norm_text for w in SOCIAL_WHITELIST)
        for fam, spec in LEX.items():
            fam_hits = []
            for w in spec.get("palabras", []):
                nw = norm_basic(w)
                if " " in nw:
                    if nw in norm_text or fuzzy_contains(norm_text, nw, 0.88):
                        fam_hits.append(w)
                else:
                    if nw in toks:
                        fam_hits.append(w)
                    elif fuzzy_contains(norm_text, nw, 0.9):
                        fam_hits.append(w)
            for pat in spec.get("patrones", []):
                try:
                    rx = re.compile(pat)
                    if rx.search(norm_text): fam_hits.append(f"pat:{pat}")
                except re.error:
                    pass
            if fam == "FACILITACION":
                if FAC_EVENT.search(norm_text) and (evento_presente or EVENT_WORDS.search(norm_text)):
                    fam_hits.append("facilitacion+evento")
            if fam_hits:
                categorias.append(fam); frases.extend(fam_hits); contrib[fam] = contrib.get(fam,0.0)+spec["peso"]
        if "OFUSCACION" in contrib and evento_presente:
            contrib["OFUSCACION"] += 1.2; frases.append("ofuscacion+evento")
        if social_hit and categorias == ["REEMBOLSO_DUDOSO"]:
            contrib["REEMBOLSO_DUDOSO"] = max(0.5, contrib["REEMBOLSO_DUDOSO"] - 0.7); frases.append("contexto_social")
        return categorias, frases, contrib, evento_presente

    def nivel(score):
        if score >= 6.5: return "CRITICO"
        if score >= 4.2: return "ALTO"
        if score >= 2.6: return "MEDIO"
        return "BAJO"

    raw_series = df[col_texto].fillna("")
    norm_series = raw_series.apply(norm_basic)
    toks_series = norm_series.apply(tokens)

    sims_per_row = [None]*len(df)
    if vectorizer is not None:
        from sklearn.metrics.pairwise import cosine_similarity
        from scipy.sparse import vstack
        Xq = vectorizer.transform(norm_series.tolist())
        fams = list(cat_centroids.keys())
        if fams:
            C = vstack([cat_centroids[f] for f in fams])
            S = cosine_similarity(Xq, C)
            for i in range(len(df)):
                row = {fams[j]: float(S[i, j]) for j in range(len(fams))}
                sims_per_row[i] = dict(sorted(row.items(), key=lambda kv: kv[1], reverse=True)[:return_similitudes_top])

    conceptos=[]; niveles=[]; puntos=[]; cats_all=[]; frases_all=[]; vag_all=[]; emo_all=[]; eventos=[]; sims_all=[]
    for i, raw in enumerate(raw_series):
        norm_t = norm_series.iat[i]; toks = toks_series.iat[i]
        vag = vaguedad(toks); emo = 1 if has_emoji(raw) else 0
        cats, frases, contrib, evento = match_families(norm_t, toks)
        if any(p in norm_t for p in ["gracias mil","favorzote","te debo la vida","eres el mejor","millones de gracias","increible","increíble","salvaste","confio en ti","confío en ti"]):
            emo += 1; contrib["EMOCIONAL"] = contrib.get("EMOCIONAL", 0.0) + 0.9; cats = list(sorted(set(cats + ["EMOCIONAL"]))); frases.append("emocional_lex")
        score = sum(contrib.values())
        if any(c in cats for c in ["REEMBOLSO_DUDOSO","NOMINA_PARALELA","PRESTAMO"]):
            score += min(1.0, vag) * 0.8
            if vag >= 0.7: frases.append("vaguedad_alta")
        if emo > 0: score += 0.3; frases.append("emoji")
        if "DEIXIS" in cats and len(cats) > 1: score += 0.6; frases.append("deixis+otra")
        if has_rel:
            r = str(df[col_relacion].iat[i]).lower()
            if "manager" in r and any(c in cats for c in ["SOBORNO","EXTORSION","OFUSCACION","FACILITACION","NOMINA_PARALELA","REEMBOLSO_DUDOSO","PRESTAMO","COI_RELACIONAL","FAVORES_SEXUALES"]):
                score += 0.8; frases.append("jerarquia")
        if sims_per_row[i] is not None:
            for fam, sim in sims_per_row[i].items():
                if sim > 0.18:
                    peso_extra = 3.6 if fam=="FAVORES_SEXUALES" else 1.0
                    score += sim * (0.9 * peso_extra)
            sims_all.append(sims_per_row[i])
        else:
            sims_all.append({})
        if cats and all(c in {"DINERO_SLANG","EMOCIONAL","DEIXIS"} for c in cats): score = min(score, 2.0)
        fam_top = max(contrib.items(), key=lambda kv: kv[1])[0] if contrib else "NINGUNO"
        concepto = "" if fam_top=="NINGUNO" else fam_top
        conceptos.append(concepto); niveles.append(nivel(score)); puntos.append(round(float(score),3))
        cats_all.append(sorted(set(cats))); frases_all.append(sorted(set(frases)))
        vag_all.append(round(vag,3)); emo_all.append(int(emo)); eventos.append(bool(evento))

    out = df.copy()
    out["concepto sospechoso"] = conceptos
    out["nivel de riesgo"] = niveles
    out["riesgo_puntos"] = puntos
    out["categorias_detectadas"] = cats_all
    out["frases_coincidentes"] = frases_all
    out["vaguedad"] = vag_all
    out["emocion"] = emo_all
    out["evento_corporativo"] = eventos
    out["similitudes_por_categoria"] = sims_all
    return out
