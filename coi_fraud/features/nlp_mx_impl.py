# (Función completa incluida; ver categoría FAVORES_SEXUALES agregada)
from typing import Optional

def nlp_mx_etiquetar_transacciones_pro(
    df,
    col_texto,
    col_relacion: Optional[str] = None,
    use_embeddings: bool = True,
    return_similitudes_top: int = 5,
):
    import html
    import math
    import re
    import unicodedata
    from difflib import SequenceMatcher

    import pandas as pd

    if col_texto not in df.columns:
        raise ValueError(f"La columna '{col_texto}' no existe en el DataFrame.")
    has_rel = col_relacion in df.columns if col_relacion else False

    ZWSP_PATTERN = re.compile(r"[\u200B-\u200D\uFEFF]")
    EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]")
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
    ACCOUNT_PATTERN = re.compile(r"\b\d{3,}-?\d{2,}\b")
    MULTI_SPACE_PATTERN = re.compile(r"\s+")
    REPETITION_PATTERN = re.compile(r"(.)\1{2,}")
    ACCENT_MAP = str.maketrans("áéíóúüñÁÉÍÓÚÜÑ", "aeiouunAEIOUUN")
    LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "@": "a", "!": "i"}

    def norm_basic(s: str) -> str:
        s = "" if s is None else str(s)
        s = ZWSP_PATTERN.sub("", s)
        s = html.unescape(s)
        s = HTML_TAG_PATTERN.sub(" ", s)
        s = URL_PATTERN.sub(" url ", s)
        s = unicodedata.normalize("NFKC", s)
        s = s.translate(ACCENT_MAP).lower().strip()
        s = "".join(LEET_MAP.get(ch, ch) for ch in s)
        s = ACCOUNT_PATTERN.sub(" cuenta ", s)
        s = REPETITION_PATTERN.sub(r"\1\1", s)
        s = MULTI_SPACE_PATTERN.sub(" ", s)
        return s

    def tokens(s: str):
        return re.findall(r"[a-z0-9]+", s)

    def has_emoji(raw: str) -> bool:
        return bool(EMOJI_PATTERN.search(raw or ""))

    LEX = {
        "SOBORNO": {
            "peso": 3.4,
            "palabras": [
                "mordida",
                "moche",
                "mochada",
                "coima",
                "aceitar",
                "engrasar",
                "chayote",
                "agradecimiento",
                "detalle",
                "regalito",
                "incentivo",
                "comision",
                "comisión",
                "bono especial",
                "por tu gestion",
                "por tu gestión",
                "por el paro",
                "por sacar la chamba",
                "palomear",
                "para el chesco",
                "para las cocas",
                "para el cafe",
                "para la gasolina",
                "por fuera",
                "dar aceite",
                "reciprocidad",
                "agradecimiento especial",
                "pa la firma",
                "propina fuerte",
                "sobre agradecimiento",
                "regalo premium",
                "caja de whisky",
                "ticket vip",
                "compensacion",
                "compensación",
                "brown envelope",
                "sweetener",
                "lubricar proceso",
                "aceite extra",
            ],
            "patrones": [
                r"\b(agradecimiento|detalle|incentivo|comisi[oó]n|bono|mordida|moche|coima|engras|aceit|chayote|propina|sweetener|compensaci[oó]n)\b.*\b(aprob|firm|palome|autoriza|asigna|libera|licitaci[oó]n)\w*",
                r"\b(por|para)\b.*\b(aprob|firm|palome|autoriza|asigna|libera|valid)\w*",
            ],
            "seeds": [
                "agradecimiento por aprobar la orden de compra",
                "incentivo por firmar contrato",
                "detalle para que se libere la orden",
                "mordida para autorizar proveedor",
                "moche por cerrar el trato",
                "por fuera para agilizar la firma",
                "regalo premium para acelerar licitacion",
                "bono especial por tu gestion en compras",
                "agradecimiento especial por liberar pago",
                "caja de whisky por la adjudicacion",
                "sweetener para asignar proveedor preferente",
                "brown envelope for contract approval",
            ],
        },
        "EXTORSION": {
            "peso": 3.1,
            "palabras": [
                "cuota",
                "derecho de piso",
                "aportacion",
                "aportación",
                "cooperacion",
                "cooperación",
                "coperacion",
                "alinearse",
                "afloja",
                "no te cierres",
                "no me quemes",
                "para que no haya problema",
                "no te atraso",
                "no te atrase",
                "mocharse",
                "mochate",
                "pago obligatorio",
                "la cuota de siempre",
                "no digas nada",
                "proteccion",
                "protección",
                "maletin",
                "cuota sindicato",
                "ayuda mensual",
                "apoyo seguridad",
            ],
            "patrones": [
                r"\b(cuota|aportaci[oó]n|cooperaci[oó]n|derecho de piso|protecci[oó]n)\b.*\b(evaluaci[oó]n|proyecto|asignaci[oó]n|auditor[ií]a)\b",
                r"(para que no haya problema|no te atraso|no te atrase|para que sigas tranquilo|para que no los visite la auditoria)",
            ],
            "seeds": [
                "cuota mensual para que no haya problema con tu evaluación",
                "aportación para no atrasar el proyecto",
                "derecho de piso para evitar problemas",
                "proteccion a cambio de que liberes la orden",
                "maletin mensual para mantener el contrato",
            ],
        },
        "OFUSCACION": {
            "peso": 2.9,
            "palabras": [
                "sin cfdi",
                "sin factura",
                "no timbrar",
                "no timbres",
                "por fuera",
                "en sobre",
                "bajo el agua",
                "discretito",
                "discreto",
                "por la sombrita",
                "off the record",
                "ajuste manual",
                "sin registro",
                "sin evidencia",
                "gasto general",
                "servicios varios",
                "consulta rapida",
                "sin soporte",
            ],
            "patrones": [
                r"(sin\s+cfdi|sin\s+factura|no\s+timbrar|no\s+timbres|por\s+fuera|en\s+sobre|bajo\s+el\s+agua|off\s+the\s+record|sin\s+soporte|gasto\s+general)",
            ],
            "seeds": [
                "pago sin cfdi por fuera",
                "en sobre y sin factura",
                "off the record sin timbrar",
                "ajuste manual sin soporte",
                "servicios varios sin detalle",
            ],
        },
        "FACILITACION": {
            "peso": 2.3,
            "palabras": [
                "agilizar",
                "destrabar",
                "desatorar",
                "prioridad",
                "darle salida",
                "liberar",
                "gestion",
                "gestión",
                "tramite",
                "trámite",
                "fast track",
                "greenlight",
                "palomear",
                "go live",
                "express",
                "prioritario",
                "pase directo",
                "fastlane",
                "puerta trasera",
                "prioridad absoluta",
                "sin fila",
                "vía rapida",
                "vip",
            ],
            "patrones": [],
            "eventos": [
                "oc",
                "po",
                "orden de compra",
                "licitacion",
                "licitación",
                "alta proveedor",
                "proveedor",
                "contrato",
                "firma",
                "cotizacion",
                "cotización",
                "renovacion",
                "renovación",
                "anticipado",
            ],
            "seeds": [
                "pago para agilizar la orden de compra",
                "gestión para liberar alta de proveedor",
                "fast track del contrato",
                "pase directo para aprobar cotizacion",
                "prioridad absoluta para liberar oc",
            ],
        },
        "NOMINA_PARALELA": {
            "peso": 1.9,
            "palabras": [
                "bono",
                "comision",
                "comisión",
                "incentivo",
                "premio",
                "gratificacion",
                "gratificación",
                "compensacion",
                "compensación",
                "bonus",
                "gratifica",
                "extra nomina",
                "fuera de nomina",
                "nomina sombra",
                "sobre amarillito",
                "bolsa aparte",
                "honorario recurrente",
            ],
            "patrones": [],
            "seeds": [
                "bono especial fuera de nómina",
                "comisión por el contrato pagada por fuera",
                "nomina sombra para gerencia",
                "honorario recurrente extra nomina",
            ],
        },
        "REEMBOLSO_DUDOSO": {
            "peso": 1.7,
            "palabras": [
                "reembolso",
                "reembolsos",
                "viaticos",
                "viáticos",
                "gastos",
                "varios",
                "servicio",
                "servicios",
                "material",
                "caja chica",
                "consumo",
                "tickets",
                "uber",
                "taxi",
                "hotel boutique",
                "upgrade",
                "spa",
                "gastos personales",
                "gift card",
                "restaurant premium",
                "boletos vip",
                "gasto corporativo",
            ],
            "patrones": [],
            "seeds": [
                "reembolso gastos sin detalle",
                "viáticos varios",
                "gastos servicio",
                "reembolso de spa sin comprobante",
                "viaticos hotel boutique lujo",
            ],
        },
        "PRESTAMO": {
            "peso": 1.8,
            "palabras": [
                "prestamo",
                "préstamo",
                "adelanto",
                "abono",
                "saldo",
                "liquidar",
                "paguitos",
                "en partes",
                "depositito",
                "pendiente mensual",
                "lo del",
                "transferencia apoyo",
                "cubrir deuda",
                "anticipo",
                "micro prestamo",
                "prestamo puente",
                "salvar tarjeta",
            ],
            "patrones": [r"\blo del\s+\d{1,2}\b", r"\banticipo\s+nomina\b", r"\bprestamo\s+personal\b"],
            "seeds": [
                "préstamo personal en partes",
                "abono del 25",
                "pendiente mensual del préstamo",
                "anticipo nomina gerente",
                "micro prestamo para salir del mes",
            ],
        },
        "COI_RELACIONAL": {
            "peso": 1.8,
            "palabras": [
                "compadre",
                "comadre",
                "primo",
                "sobrino",
                "carnal",
                "cuate",
                "amigazo",
                "de confianza",
                "palanca",
                "conecte",
                "el de siempre",
                "ya sabes quien",
                "ya sabes quién",
                "hermana",
                "tio",
                "familia",
                "mi socio",
                "mi recomendado",
                "ahijado",
                "padrino",
                "mi gente",
                "mi gallo",
            ],
            "patrones": [r"\bmi\s+(compadre|primo|ahijado|gente|gallo|socio)\b"],
            "seeds": [
                "apoyo para compadre de compras",
                "conecte de proveedor",
                "recomendado de la familia",
                "proveedor primo gerente",
                "mi socio recibira pago",
            ],
        },
        "DINERO_SLANG": {
            "peso": 1.0,
            "palabras": [
                "lana",
                "varo",
                "feria",
                "billete",
                "pasta",
                "morlacos",
                "mangos",
                "verdes",
                "2k",
                "5k",
                "kilo",
                "bolson",
                "bolsón",
                "morralla",
                "plata",
                "cash",
                "money",
                "lucas",
                "lucro",
            ],
            "patrones": [r"\b\d+\s*k\b", r"\b(?:bolsa|sobre)\s+de\s+lana\b"],
            "seeds": [
                "2k para el trámite",
                "cinco mil para el cafe",
                "lana extra para acelerar",
                "cash para la firma",
            ],
        },
        "CODIGO": {
            "peso": 1.2,
            "palabras": [
                "c-azul",
                "c-verde",
                "c-naranja",
                "px-",
                "off the record",
                "plan delta",
                "codigo amber",
                "clave sombra",
                "modo sigilo",
                "operacion oculta",
                "canal negro",
            ],
            "patrones": [r"\b[a-z]{1,3}-\d{1,4}\b", r"\bclave\s+(?:sombra|amber|oro)\b"],
            "seeds": [
                "px-9 listo",
                "c-azul ok",
                "clave sombra autorizada",
                "operacion oculta verde",
            ],
        },
        "COORDINACION_REITERADA": {
            "peso": 1.2,
            "palabras": [
                "lo de ayer",
                "lo de antier",
                "la vez pasada",
                "como quedamos",
                "igual que ayer",
                "igual que antes",
                "igual trato",
                "lo de siempre",
                "lo acostumbrado",
                "misma jugada",
                "mismo trato",
                "lo pendiente de ayer",
            ],
            "patrones": [
                r"lo\s+de\s+(ayer|siempre|antes)",
                r"(misma|mismo)\s+(jugada|trato)",
            ],
            "seeds": [
                "lo de ayer igual",
                "como quedamos",
                "misma jugada de siempre",
            ],
        },
        "ALUSION_INDIRECTA": {
            "peso": 1.0,
            "palabras": [
                "ya sabes",
                "aquello",
                "eso",
                "lo pendiente",
                "lo conversado",
                "ya quedamos",
                "lo hablado",
                "el tema",
            ],
            "patrones": [
                r"ya\s+sabes",
                r"lo\s+pendiente",
            ],
            "seeds": [
                "ya sabes que hacer",
                "lo pendiente queda igual",
            ],
        },
        "EMOCIONAL": {
            "peso": 0.9,
            "palabras": [
                "gracias mil",
                "favorzote",
                "te debo la vida",
                "eres el mejor",
                "millones de gracias",
                "increible",
                "increíble",
                "salvaste",
                "confio en ti",
                "confío en ti",
                "mil gracias",
                "apoyo infinito",
                "eres un crack",
                "salvador",
                "idolo",
                "heroina",
                "heroína",
                "gracias por tanto",
                "gracias totales",
                "agradezco mucho",
            ],
            "patrones": [],
            "seeds": [
                "gracias mil por el paro",
                "favorzote con lo de la firma",
                "eres un crack por liberar",
                "heroina por sacar contrato",
            ],
        },
        "FAVORES_SEXUALES": {
            "peso": 3.6,
            "palabras": [
                "favor sexual",
                "sexo",
                "salida intima",
                "salida íntima",
                "cariño especial",
                "cita privada",
                "encuentro",
                "acompaname",
                "acompáname",
                "acompanamiento",
                "trato especial",
                "detallito especial",
                "algo mas",
                "algo más",
                "noche juntos",
                "masaje privado",
                "after office privado",
                "cena romantica",
                "vino en suite",
                "tanga",
                "tangas",
                "lenceria",
                "lencería",
                "tus tangas",
                "elefant",
            ],
            "patrones": [
                r"(favor|salida|cita|encuentro).{0,10}(intim|privad|personal)",
                r"(carin(?:o|\xF3)|cari\xF1o|mimos?).{0,10}(extra|especial)",
                r"(a\s+cambio\s+de).{0,12}(salida|cita|favor)",
                r"(trato\s+especial).{0,20}(firma|aproba|libera|contrato|orden)",
                r"(after\s+office).{0,15}(privado|suite)",
            ],
            "seeds": [
                "favor sexual por firma",
                "salida íntima a cambio de aprobar",
                "cita privada para liberar la orden",
                "trato especial por el contrato",
                "noche juntos por adjudicacion",
            ],
        },
        "AGASAJOS_SOCIALES": {
            "peso": 2.0,
            "palabras": [
                "cerveza",
                "chela",
                "chelas",
                "tragos",
                "pomo",
                "after",
                "aftercito",
                "fiesta",
                "fiestecita",
                "antro",
                "karaoke",
                "botella",
                "tequila",
                "mezcal",
                "brindis",
                "cena bar",
                "tapas",
            ],
            "patrones": [
                r"(after|fiesta|antro|karaoke)",
                r"(botella|pomo|tragos?).{0,10}(vip|premium)?",
            ],
            "seeds": [
                "fiesta con pomos para celebrar aprobacion",
                "chelas despues de liberar contrato",
                "after en el antro por la firma",
            ],
        },
        "DETALLE_PERSONAL": {
            "peso": 2.2,
            "palabras": [
                "regalo",
                "regalito",
                "detallito",
                "detallazo",
                "detallito especial",
                "detalle personal",
                "detalle para ti",
                "te compras algo bonito",
                "tecomprasalgobonito",
                "tecomprasalgobonit",
                "te compras algo bonit",
                "reglo",
            ],
            "patrones": [
                r"te\s*compras?\s*algo\s*bonit",
                r"regal(?:o|ito)",
            ],
            "seeds": [
                "te compras algo bonito por el apoyo",
                "regalito especial por liberar contrato",
                "detalle personal por tu ayuda",
            ],
        },
        "REGALOS_LUJO": {
            "peso": 2.8,
            "palabras": [
                "bolsa louis",
                "reloj rolex",
                "viaje cancun",
                "boletos vip",
                "concierto vip",
                "spa de lujo",
                "cena gourmet",
                "suite presidencial",
                "vino premium",
                "auto demo",
                "yate",
                "joyeria",
                "joyería",
                "ticket premier",
                "regalo lujo",
                "detallazo",
            ],
            "patrones": [
                r"(viaje|cena|boletos|suite|hotel|spa|yate|reloj|auto).{0,20}(vip|luj[oa]|premium|exclusivo)",
            ],
            "seeds": [
                "viaje cancun vip para agradecer",
                "reloj rolex para aprobar contrato",
                "suite presidencial por la licitacion",
                "cena gourmet a cambio de autorizacion",
            ],
        },
        "CONFLICTO_INTERES_FAMILIAR": {
            "peso": 3.0,
            "palabras": [
                "esposo",
                "esposa",
                "hermano",
                "hermana",
                "sobrina",
                "sobrino",
                "yerno",
                "nuera",
                "familia",
                "familiar",
                "pariente",
                "mi pareja",
                "mi esposa",
                "mi esposo",
                "mi hijo",
                "mi hija",
                "cunyado",
                "cuñado",
                "tio",
                "tia",
                "suegro",
                "suegra",
                "parentesco",
                "mis padres",
                "nuestros hijos",
            ],
            "patrones": [
                r"mi\s+(esposo|esposa|hermano|hijo|hija|pareja|familia)",
                r"(contrato|orden).{0,20}(para\s+mi\s+(?:familia|primo|hijo|hermana))",
            ],
            "seeds": [
                "orden para mi esposo proveedor",
                "contrato asignado a mi hermana",
                "pago para empresa de mi hijo",
                "factura familiar autorizada",
            ],
        },
        "VIATICOS_LUJOSOS": {
            "peso": 2.5,
            "palabras": [
                "viaticos cancun",
                "viaticos dubai",
                "hotel cinco estrellas",
                "first class",
                "primera clase",
                "upgrade suite",
                "spa ejecutivo",
                "tour vip",
                "viaje familiar",
                "boleto business",
            ],
            "patrones": [r"vi[aá]ticos?.{0,15}(vip|premium|luj[oa]|business)"],
            "seeds": [
                "viaticos dubai aprobados",
                "hotel cinco estrellas para reunion",
                "upgrade suite viaticos",
                "viaje familiar con viaticos",
            ],
        },
        "FACTURACION_SIMULADA": {
            "peso": 3.2,
            "palabras": [
                "empresa fachada",
                "compra fantasma",
                "servicio duplicado",
                "factura espejo",
                "proveedor fantasma",
                "proveedor nuevo sin historial",
                "facturacion cruzada",
                "servicio inventado",
                "honorario ficticio",
                "contrato simulado",
                "cotizacion espejo",
            ],
            "patrones": [
                r"(factura|servicio|compra).{0,15}(fantasma|simulad|espejo|duplicad|fachada)",
            ],
            "seeds": [
                "factura fantasma proveedor amigo",
                "servicio duplicado fachada",
                "empresa fachada manda factura",
                "contrato simulado para justificar",
            ],
        },
        "CONSULTORIA_FANTASMA": {
            "peso": 2.6,
            "palabras": [
                "consultoria express",
                "asesoria relampago",
                "informe copia",
                "powerpoint reusado",
                "documento generico",
                "horas no trabajadas",
                "reportes reciclados",
                "propuesta copy paste",
                "entrega simbolica",
                "entrega dummy",
            ],
            "patrones": [r"consultor[ií]a.{0,15}(fantasma|dummy|express|rel[aá]mpago)"],
            "seeds": [
                "consultoria express sin entregables",
                "asesoria fantasma para cobrar",
                "informe copia para justificar pago",
            ],
        },
        "DONATIVO_CRUZADO": {
            "peso": 2.4,
            "palabras": [
                "donativo",
                "donacion",
                "donación",
                "fundacion",
                "fundación",
                "apoyo campaña",
                "campana",
                "campaña",
                "sponsor",
                "patrocinio",
                "apoyo evento",
                "cuota politicos",
                "aportacion partido",
            ],
            "patrones": [
                r"(donativo|donaci[oó]n|patrocinio|aporte).{0,20}(camp[aá]n|fundaci[oó]n|partido|evento)",
            ],
            "seeds": [
                "donativo fundacion recomendado gerente",
                "aportacion partido desde caja chica",
                "patrocinio campaña contacto proveedor",
            ],
        },
        "PRESION_POLITICA": {
            "peso": 2.7,
            "palabras": [
                "diputado",
                "senador",
                "candidato",
                "campana",
                "campaña",
                "delegado",
                "regidor",
                "secretario",
                "partido",
                "comite",
                "comité",
                "representante",
                "influencia politica",
                "palanca politica",
            ],
            "patrones": [
                r"(candidato|senador|diputado|partido|delegado|camp[aá]n).{0,20}(apoyo|favor|contrato|evento)",
            ],
            "seeds": [
                "apoyo candidato libera contrato",
                "favor partido para asignacion",
                "palanca politica exige pago",
            ],
        },
    }

    SOCIAL_WHITELIST = {
        "cumple",
        "despedida",
        "baby shower",
        "boda",
        "vaquita",
        "coperacha",
        "pastel",
        "regalo despedida",
    }

    raw_series = df[col_texto].fillna("").astype(str)
    norm_series = raw_series.apply(norm_basic)
    toks_series = norm_series.apply(tokens)

    vectorizer = None
    cat_centroids = {}
    cat_centroids_matrix = None
    fams_for_similarity: list[str] = []
    embedding_expanded_terms: dict[str, set[str]] = {fam: set() for fam in LEX}
    if use_embeddings:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            lexical_docs: list[str] = []
            y: list[str] = []
            base_terms_per_fam: dict[str, set[str]] = {fam: set() for fam in LEX}
            augmentation_contexts = [
                "licitacion urgente",
                "proveedor favorito",
                "contrato directo",
                "evaluacion trimestral",
                "cierre anual",
                "campana interna",
                "auditoria sorpresa",
                "servicio premium",
                "compra extraordinaria",
                "evento corporativo",
            ]
            for fam, spec in LEX.items():
                seen_terms: set[str] = set()
                for s in spec.get("seeds", []):
                    base = norm_basic(s)
                    if base and base not in seen_terms:
                        lexical_docs.append(base)
                        y.append(fam)
                        seen_terms.add(base)
                        base_terms_per_fam[fam].add(base)
                    for ctx in augmentation_contexts:
                        combo = norm_basic(f"{s} {ctx}")
                        if combo and combo not in seen_terms:
                            lexical_docs.append(combo)
                            y.append(fam)
                            seen_terms.add(combo)
                for w in spec.get("palabras", []):
                    base = norm_basic(w)
                    if base and base not in seen_terms:
                        lexical_docs.append(base)
                        y.append(fam)
                        seen_terms.add(base)
                        base_terms_per_fam[fam].add(base)
                    if " " not in w:
                        doubled = norm_basic(f"{w} urgente")
                        if doubled and doubled not in seen_terms:
                            lexical_docs.append(doubled)
                            y.append(fam)
                            seen_terms.add(doubled)
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
            extra_docs: list[str] = []
            seen_extra: set[str] = set()
            unique_norm_texts = list(dict.fromkeys(norm_series.tolist()))
            for text in unique_norm_texts:
                if not text or text in seen_extra:
                    continue
                extra_docs.append(text)
                seen_extra.add(text)
                if len(extra_docs) >= 4000:
                    break
            fit_corpus = lexical_docs + extra_docs
            if not fit_corpus:
                raise ValueError("No hay corpus para inicializar embeddings.")
            vectorizer.fit(fit_corpus)
            X = vectorizer.transform(lexical_docs)
            fam_to: dict[str, list] = {}
            for fam, row in zip(y, X):
                fam_to.setdefault(fam, []).append(row)
            for fam, mats in fam_to.items():
                cat_centroids[fam] = mats[0] if len(mats) == 1 else sum(mats) / len(mats)
            if cat_centroids:
                from scipy.sparse import vstack

                fams_for_similarity = list(cat_centroids.keys())
                cat_centroids_matrix = vstack([cat_centroids[f] for f in fams_for_similarity])

                candidate_segments: list[str] = []
                seen_segments: set[str] = set()
                MAX_SEGMENTS = 12000
                for text in unique_norm_texts:
                    if len(candidate_segments) >= MAX_SEGMENTS:
                        break
                    words = [w for w in text.split() if len(w) >= 3]
                    if not words:
                        continue
                    for size in (1, 2, 3):
                        if len(words) < size:
                            continue
                        for idx in range(len(words) - size + 1):
                            segment = " ".join(words[idx : idx + size])
                            if len(segment) < 4 or segment in seen_segments:
                                continue
                            seen_segments.add(segment)
                            candidate_segments.append(segment)
                            if len(candidate_segments) >= MAX_SEGMENTS:
                                break
                        if len(candidate_segments) >= MAX_SEGMENTS:
                            break
                if candidate_segments:
                    from sklearn.metrics.pairwise import cosine_similarity

                    X_candidates = vectorizer.transform(candidate_segments)
                    if X_candidates.nnz:
                        sims_segments = cosine_similarity(X_candidates, cat_centroids_matrix)
                        for seg_idx, sims_row in enumerate(sims_segments):
                            for fam_idx, sim in enumerate(sims_row):
                                if sim >= 0.32:
                                    fam = fams_for_similarity[fam_idx]
                                    seg = candidate_segments[seg_idx]
                                    if seg not in base_terms_per_fam[fam]:
                                        embedding_expanded_terms[fam].add(seg)
        except Exception:
            vectorizer = None
            cat_centroids = {}
            cat_centroids_matrix = None
            fams_for_similarity = []
            embedding_expanded_terms = {fam: set() for fam in LEX}

    def vaguedad(tokens_list):
        VAG_TERMS = {
            "apoyo",
            "gasto",
            "gastos",
            "reembolso",
            "varios",
            "servicio",
            "servicios",
            "material",
            "proyecto",
            "viaticos",
            "viatico",
            "viáticos",
            "bono",
            "comision",
            "comisión",
            "concepto",
            "detalle",
            "pago",
            "factura",
        }
        if not tokens_list:
            return 1.0
        v = sum(1 for t in tokens_list if t in VAG_TERMS) / max(1, len(tokens_list))
        if len(tokens_list) <= 2:
            v = max(v, 0.8)
        return float(min(1.0, max(0.0, v)))

    def sentimiento(tokens_list):
        if not tokens_list:
            return 0.0
        sentiment_pos = {
            "agradecido",
            "agradecida",
            "gracias",
            "increible",
            "increíble",
            "excelente",
            "perfecto",
            "feliz",
            "contento",
            "contenta",
            "exitoso",
            "maravilloso",
            "mil",
            "aprecio",
            "apreciamos",
            "buenisimo",
            "buenísimo",
            "fantastico",
            "fantástico",
        }
        sentiment_neg = {
            "molesto",
            "molesta",
            "presion",
            "presión",
            "urgente",
            "exigencia",
            "amenaza",
            "amenazo",
            "estres",
            "estrés",
            "queja",
            "problema",
            "reclamo",
            "ultimatum",
            "ultimátum",
            "complica",
            "complicado",
            "estresado",
            "estresada",
        }
        pos = sum(1 for t in tokens_list if t in sentiment_pos)
        neg = sum(1 for t in tokens_list if t in sentiment_neg)
        score = (pos - neg) / max(1, len(tokens_list))
        return float(max(-1.0, min(1.0, score)))

    def fuzzy_contains(haystack: str, needle: str, thresh=0.86) -> bool:
        if " " in needle and needle in haystack:
            return True
        if len(needle) < 4:
            return needle in haystack
        max_window = max(3, len(needle) + 6)
        for idx, m in enumerate(re.finditer(r"[a-z0-9][a-z0-9\s]{0,%d}" % max_window, haystack)):
            if idx >= 50:
                break
            seg = m.group(0)
            if SequenceMatcher(None, seg[: len(needle) + 3], needle).ratio() >= thresh:
                return True
        if len(haystack) > 600:
            haystack = haystack[:600]
        return SequenceMatcher(None, haystack, needle).ratio() >= (thresh + 0.03)

    FAC_EVENT = re.compile(
        r"(agiliz|destrab|desator|prioridad|darle salida|liber|gesti[oó]n|tramite|tr[aá]mite|fast[- ]?track|greenlight|palomear|fastlane|vip)"
    )
    EVENT_WORDS = re.compile(
        r"(?:\boc\b|\bpo\b|orden de compra|licitaci[oó]n|alta proveedor|proveedor|contrato|firma|cotizaci[oó]n|renovaci[oó]n)"
    )

    def match_families(norm_text, toks):
        categorias = []
        frases = []
        contrib: dict[str, float] = {}
        evento_presente = bool(EVENT_WORDS.search(norm_text))
        social_hit = any(w in norm_text for w in SOCIAL_WHITELIST)
        for fam, spec in LEX.items():
            fam_hits = []
            embed_bonus = 0.0
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
            emb_hits = []
            for emb in embedding_expanded_terms.get(fam, ()):  # términos sugeridos por embeddings
                if emb in norm_text or fuzzy_contains(norm_text, emb, 0.92):
                    emb_hits.append(emb)
            if emb_hits:
                fam_hits.extend(f"emb:{emb}" for emb in emb_hits)
                embed_bonus = spec["peso"] * (0.45 if len(emb_hits) == 1 else 0.7)
            for pat in spec.get("patrones", []):
                try:
                    rx = re.compile(pat)
                    if rx.search(norm_text):
                        fam_hits.append(f"pat:{pat}")
                except re.error:
                    pass
            if fam == "FACILITACION":
                if FAC_EVENT.search(norm_text) and (evento_presente or EVENT_WORDS.search(norm_text)):
                    fam_hits.append("facilitacion+evento")
            if fam_hits:
                categorias.append(fam)
                frases.extend(fam_hits)
                contrib[fam] = contrib.get(fam, 0.0) + spec["peso"] + embed_bonus
        if "OFUSCACION" in contrib and evento_presente:
            contrib["OFUSCACION"] += 1.2
            frases.append("ofuscacion+evento")
        if social_hit and categorias == ["REEMBOLSO_DUDOSO"]:
            contrib["REEMBOLSO_DUDOSO"] = max(0.5, contrib["REEMBOLSO_DUDOSO"] - 0.7)
            frases.append("contexto_social")
        return categorias, frases, contrib, evento_presente

    def nivel(score):
        if score >= 6.5:
            return "CRITICO"
        if score >= 4.2:
            return "ALTO"
        if score >= 2.6:
            return "MEDIO"
        return "BAJO"

    sims_per_row = [None] * len(df)
    if vectorizer is not None and cat_centroids_matrix is not None and fams_for_similarity:
        from sklearn.metrics.pairwise import cosine_similarity

        Xq = vectorizer.transform(norm_series.tolist())
        S = cosine_similarity(Xq, cat_centroids_matrix)
        for i in range(len(df)):
            row = {fams_for_similarity[j]: float(S[i, j]) for j in range(len(fams_for_similarity))}
            sims_per_row[i] = dict(
                sorted(row.items(), key=lambda kv: kv[1], reverse=True)[:return_similitudes_top]
            )

    conceptos = []
    niveles = []
    puntos = []
    cats_all = []
    frases_all = []
    vag_all = []
    emo_all = []
    eventos = []
    sims_all = []
    senti_all = []
    coi_scores = []

    for i, raw in enumerate(raw_series):
        norm_t = norm_series.iat[i]
        toks = toks_series.iat[i]
        vag = vaguedad(toks)
        emo = 1 if has_emoji(raw) else 0
        senti = sentimiento(toks)
        cats, frases, contrib, evento = match_families(norm_t, toks)
        if any(
            p in norm_t
            for p in [
                "gracias mil",
                "favorzote",
                "te debo la vida",
                "eres el mejor",
                "millones de gracias",
                "increible",
                "increíble",
                "salvaste",
                "confio en ti",
                "confío en ti",
            ]
        ):
            emo += 1
            contrib["EMOCIONAL"] = contrib.get("EMOCIONAL", 0.0) + 0.9
            cats = list(sorted(set(cats + ["EMOCIONAL"])))
            frases.append("emocional_lex")
        score = sum(contrib.values())
        if any(c in cats for c in ["REEMBOLSO_DUDOSO", "NOMINA_PARALELA", "PRESTAMO"]):
            score += min(1.0, vag) * 0.8
            if vag >= 0.7:
                frases.append("vaguedad_alta")
        if emo > 0:
            score += 0.3
            frases.append("emoji")
        if any(c in {"COORDINACION_REITERADA", "ALUSION_INDIRECTA"} for c in cats) and len(cats) > 1:
            score += 0.6
            frases.append("coordinacion_vaga")
        if has_rel:
            r = str(df[col_relacion].iat[i]).lower()
            if "manager" in r and any(
                c
                in {
                    "SOBORNO",
                    "EXTORSION",
                    "OFUSCACION",
                    "FACILITACION",
                    "NOMINA_PARALELA",
                    "REEMBOLSO_DUDOSO",
                    "PRESTAMO",
                    "COI_RELACIONAL",
                    "FAVORES_SEXUALES",
                    "REGALOS_LUJO",
                    "AGASAJOS_SOCIALES",
                    "DETALLE_PERSONAL",
                    "CONFLICTO_INTERES_FAMILIAR",
                    "FACTURACION_SIMULADA",
                    "CONSULTORIA_FANTASMA",
                    "DONATIVO_CRUZADO",
                    "PRESION_POLITICA",
                }
                for c in cats
            ):
                score += 0.8
                frases.append("jerarquia")
        sentiment_boost = 0.0
        if senti > 0.25:
            sentiment_boost = min(0.6, senti * 1.4)
            frases.append("sentimiento_positivo_excesivo")
        elif senti < -0.2:
            sentiment_boost = min(0.7, abs(senti) * 1.5)
            frases.append("sentimiento_negativo_presion")
        if sims_per_row[i] is not None:
            for fam, sim in sims_per_row[i].items():
                if sim > 0.18:
                    peso_extra = 3.6 if fam == "FAVORES_SEXUALES" else 1.0
                    score += sim * (0.9 * peso_extra)
            sims_all.append(sims_per_row[i])
        else:
            sims_all.append({})
        if cats and all(
            c in {"DINERO_SLANG", "EMOCIONAL", "COORDINACION_REITERADA", "ALUSION_INDIRECTA"}
            for c in cats
        ):
            score = min(score, 2.0)
        fam_top = max(contrib.items(), key=lambda kv: kv[1])[0] if contrib else "NINGUNO"
        concepto = "" if fam_top == "NINGUNO" else fam_top
        conceptos.append(concepto)
        niveles.append(nivel(score))
        puntos.append(round(float(score), 3))
        cats_all.append(sorted(set(cats)))
        frases_all.append(sorted(set(frases)))
        vag_all.append(round(vag, 3))
        emo_all.append(int(emo))
        eventos.append(bool(evento))
        senti_all.append(round(float(senti), 3))
        coi_score = float(score + sentiment_boost + (vag * 0.6) + (abs(senti) * 0.5))
        coi_scores.append(round(coi_score, 3))

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
    out["sentimiento"] = senti_all
    out["score_probable_coi"] = coi_scores
    return out
