"""Synthetic data generation utilities.

This module provides helpers to build highly diverse transactional datasets
that comply with the minimum schema required by :func:`coi_fraud.run_pipeline`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Persona:
    """Represents a synthetic employee with rich metadata."""

    user_id: str
    full_name: str
    job_title: str
    age: int
    state: str
    tenure_years: float


_FIRST_NAMES = [
    "Abril",
    "Beatriz",
    "Camila",
    "Diego",
    "Elena",
    "Fabiola",
    "Gerardo",
    "Hugo",
    "Itzel",
    "Javier",
    "Katia",
    "Lourdes",
    "Manuel",
    "Nidia",
    "Octavio",
    "Paola",
    "Quique",
    "Rafael",
    "Sara",
    "Tania",
    "Uriel",
    "Valeria",
    "Wendy",
    "Ximena",
    "Yahir",
    "Zaira",
]

_LAST_NAMES = [
    "Alvarado",
    "Bautista",
    "Cortés",
    "Delgado",
    "Espinoza",
    "Fernández",
    "García",
    "Hernández",
    "Ibarra",
    "Juárez",
    "Kuri",
    "López",
    "Mendoza",
    "Navarro",
    "Ortega",
    "Pineda",
    "Quintana",
    "Ramírez",
    "Salinas",
    "Torres",
    "Uribe",
    "Vega",
    "Wong",
    "Xicoténcatl",
    "Yáñez",
    "Zúñiga",
]

_JOBS = [
    "Analista de Datos",
    "Gerente de Compras",
    "Coordinador de Nómina",
    "Especialista en Ciberseguridad",
    "Director de Operaciones",
    "Consultor ESG",
    "Ingeniera de Campo",
    "Desarrolladora Backend",
    "Investigador de Fraude",
    "Product Owner",
    "Arquitecta Cloud",
    "Abogada Corporativa",
    "Diseñador UX",
    "Supervisor de Planta",
    "Agente de Cobranza",
    "Scrum Master",
    "Químico de Laboratorio",
    "Coordinador de Logística",
    "Analista de Riesgos",
    "Ejecutiva de Ventas",
    "Especialista en Nómina Global",
    "Psicóloga Organizacional",
    "Arquitecto de Soluciones",
    "Gerente de Sostenibilidad",
    "Ingeniero de DevOps",
    "Responsable de Cumplimiento",
]

_STATES = [
    "AGS",
    "BCN",
    "BCS",
    "CAM",
    "CHH",
    "CHP",
    "CMX",
    "COA",
    "COL",
    "DUR",
    "GRO",
    "GUA",
    "HID",
    "JAL",
    "MEX",
    "MIC",
    "MOR",
    "NAY",
    "NLE",
    "OAX",
    "PUE",
    "QUE",
    "ROO",
    "SIN",
    "SLP",
    "SON",
    "TAB",
    "TAM",
    "TLA",
    "VER",
    "YUC",
    "ZAC",
]

_ORIGIN_APPS = [
    "portal_web",
    "api_partners",
    "mobile_android",
    "mobile_ios",
    "kiosk",
    "sap_bridge",
    "workday_sync",
    "conta_fiscal",
]

_SERVICE_TAGS = [
    "pago_servicios",
    "reembolso_caja_chica",
    "viatico",
    "bono_excepcional",
    "compra_tecnologia",
    "donativo",
    "proveedor_internacional",
    "consultoria",
    "suscripcion_saas",
    "material_medico",
    "marketing_digital",
    "infraestructura_nube",
    "capacitacion",
    "legal_arbitraje",
    "logistica_aduanas",
    "mantenimiento_industrial",
    "viaje_ecologico",
    "investigacion_ux",
    "prototipo_iot",
    "servicios_financieros",
    "software_libre",
    "campana_multicultural",
    "alianza_ong",
    "blockchain",
    "laboratorio_genomica",
    "arte_urbano",
    "patrocinio_deportivo",
    "inclusion_laboral",
    "educacion_stem",
    "inteligencia_competitiva",
    "auditoria_externa",
    "ciber_resiliencia",
    "energia_renovable",
    "microcredito",
    "robotica_colaborativa",
    "comercio_justo",
]

_DETAIL_TAGS = [
    "piloto-latam",
    "certificacion_iso",
    "hackathon_europeo",
    "campamento_ai",
    "mision_humanitaria",
    "alianza_estrategica",
    "reciclaje-oceanico",
    "festival_culinario",
    "expansion_agronegocio",
    "incubadora_social",
    "sprint_bimensual",
    "laboratorio-biotecnologia",
    "cumbre_solar",
    "marketplace_artesanal",
    "pasantia_internacional",
    "crowdfunding_maker",
    "co-creacion_comunidad",
    "rescate_patrones_eticos",
    "alianza_gobierno_abierto",
    "fondo_empoderamiento",
    "interoperabilidad_openbanking",
    "circuito_movilidad",
    "agenda_indigena",
    "climatologia_predictiva",
    "salud_mental",
    "programa_neuroliderazgo",
    "cafe_sustentable",
    "criptodonativo",
    "turismo_espacial",
    "realidad_mixta",
    "derechos_digitales",
    "bosque_urbano",
]

_CODE_TAGS = [
    "MX-OPS-001",
    "GDL-CX-204",
    "QRO-RPA-312",
    "MTY-FIN-778",
    "CDMX-AI-887",
    "VER-MKT-459",
    "PUE-HR-992",
    "SLP-ENG-640",
    "NLE-COMP-552",
    "BCN-CIBER-724",
    "YUC-SUST-038",
    "CHH-LOG-619",
    "BCS-LEGAL-504",
]

_LANGUAGE_SNIPPETS = [
    "urgent follow-up",
    "mise à jour",
    "ajuste inmediato",
    "relatório preliminar",
    "due diligence",
    "prueba de concepto",
    "benchmark ético",
    "observatorio ciudadano",
    "pilot de innovación",
    "sesión híbrida",
]

_BEHAVIOR_CASE_TYPES = [
    ("canal_etico", "Reporte canal ético"),
    ("publicacion_interna", "Publicación en intranet"),
    ("ticket_rrhh", "Ticket confidencial RRHH"),
    ("nota_comunidad", "Nota de comunidad"),
]

_MEXICAN_SLANG = [
    "neta que esto está gacho",
    "órale con la situación",
    "ya estuvo, banda",
    "qué onda con ese rollo",
    "se siente bien pesado, la neta",
    "aguas porque se está pasando",
    "la raza anda incómoda, wey",
]

_INAPPROPRIATE_FLAGS = [
    ("acoso verbal recurrente", "acoso"),
    ("amenaza directa de despido si no ceden", "amenaza"),
    ("hostigamiento constante por mensajes", "hostigamiento"),
    ("chantaje para conseguir favores", "chantaje"),
    ("solicitud de soborno para autorizar gastos", "soborno"),
    ("coacción para cubrir irregularidades", "coacción"),
    ("abuso de autoridad frente al equipo", "abuso"),
    ("exigencia de una 'coperación' para liberar pagos", "coperación"),
    ("presión para aportar a una coperación que encubre faltantes", "coperación"),
]

_BEHAVIOR_CONTEXTS = [
    "durante las guardias nocturnas en planta",
    "en los chats del proyecto de expansión",
    "cuando revisan los viáticos en piso",
    "en las juntas híbridas con proveedores",
    "durante las capacitaciones obligatorias",
    "en los relevos de turno del centro de soporte",
    "al cierre de los reportes trimestrales",
    "cuando piden transferir coperaciones en efectivo",
    "en las colectas improvisadas para cubrir supuestos errores",
]

_BEHAVIOR_ACTORS = [
    "personal de soporte",
    "equipo de compras",
    "colegas de logística",
    "staff de operaciones",
    "compañeras de atención a clientes",
    "ingeniería de campo",
    "analistas de cumplimiento",
]

_BEHAVIOR_REACTIONS = [
    "varias personas pidieron frenar el acoso de inmediato",
    "se documentó la amenaza y se pidió apoyo formal",
    "se levantó alerta por el hostigamiento constante",
    "se denunció el chantaje ante los líderes",
    "se rechazó el soborno y se solicitó investigación",
    "se reportó la coacción con evidencia en archivos",
    "se elevó el abuso para resguardar al equipo",
]

_BEHAVIOR_ESCALATION = [
    "se considera incidente urgente",
    "se marcó como caso crítico",
    "amerita intervención inmediata",
    "se propone activar protocolo de protección",
    "se programó seguimiento prioritario",
]

_SCENARIOS = [
    ("payroll", 1.0, 32000, 6000),
    ("micro", 0.8, 1200, 300),
    ("compliance", 1.2, 8500, 2200),
    ("intl", 1.4, 20000, 9000),
    ("crypto", 2.0, 15000, 100),
    ("rnd", 1.5, 50000, 12000),
    ("events", 0.7, 7000, 1500),
    ("logistics", 1.1, 11000, 4000),
    ("donations", 0.6, 9500, 5000),
    ("procurement", 1.3, 18000, 6500),
]


def _build_personas(rng: np.random.Generator, size: int) -> List[Persona]:
    ids = [f"P{idx:04d}" for idx in range(1, size + 1)]
    first = rng.choice(_FIRST_NAMES, size=size)
    last = rng.choice(_LAST_NAMES, size=size)
    jobs = rng.choice(_JOBS, size=size)
    ages = rng.integers(22, 65, size=size)
    states = rng.choice(_STATES, size=size)
    tenure = rng.uniform(0.1, 25.0, size=size)
    personas = [
        Persona(
            user_id=pid,
            full_name=f"{f} {l}",
            job_title=job,
            age=int(age),
            state=state,
            tenure_years=round(float(t), 2),
        )
        for pid, f, l, job, age, state, t in zip(ids, first, last, jobs, ages, states, tenure)
    ]
    return personas


def _sample_persona_metadata(persona: Persona, prefix: str) -> dict:
    return {
        f"{prefix}-nombre_completo": persona.full_name,
        f"{prefix}-puesto": persona.job_title,
        f"{prefix}-edad": persona.age,
        f"{prefix}-state_id": persona.state,
        f"{prefix}-gf_worker_hiring_date": (
            datetime.now() - timedelta(days=int(persona.tenure_years * 365.25))
        ).strftime("%Y-%m-%d"),
    }


def _build_description(rng: np.random.Generator) -> str:
    service = rng.choice(_SERVICE_TAGS)
    detail = rng.choice(_DETAIL_TAGS)
    code = rng.choice(_CODE_TAGS)
    snippet = rng.choice(_LANGUAGE_SNIPPETS)
    template = rng.choice(
        [
            "{service}; {detail}; ref {code}; {snippet}",
            "{service} | {detail} | expediente {code} | {snippet}",
            "{service} - {detail} ({code}) [{snippet}]",
        ]
    )
    return template.format(service=service, detail=detail, code=code, snippet=snippet)


def _sample_teammates(rng: np.random.Generator, pool: Sequence[str], size: int) -> str:
    teammates = rng.choice(pool, size=size, replace=False)
    return ",".join(sorted(set(teammates)))


def _sample_manager_chain(rng: np.random.Generator, pool: Sequence[str], max_depth: int = 4) -> List[str]:
    depth = rng.integers(0, max_depth + 1)
    if depth == 0:
        return []
    managers = rng.choice(pool, size=depth, replace=False)
    return list(managers)


def _scenario_amount(rng: np.random.Generator) -> float:
    """Sample an amount based on one of the predefined scenarios."""

    # ``Generator.choice`` intenta vectorizar la selección y, al recibir una
    # lista de tuplas con tipos heterogéneos, convierte todo el arreglo en
    # ``numpy.str_``. Eso provoca que ``mean`` y ``spread`` sean cadenas y que
    # la división falle. Seleccionamos el índice manualmente para conservar los
    # tipos originales de Python.
    kind, shape, mean, spread = _SCENARIOS[int(rng.integers(0, len(_SCENARIOS)))]
    scale = mean
    sigma = max(spread / mean, 0.05)
    amount = rng.lognormal(mean=np.log(scale), sigma=sigma)
    jitter = rng.normal(0, spread * 0.1)
    raw = amount + jitter
    if kind == "crypto":
        raw *= rng.choice([0.5, 1.0, 1.5, 2.5])
    if kind == "donations":
        raw = abs(raw)
    return float(np.clip(raw, 50.0, 250000.0))


def _build_behavioral_case(rng: np.random.Generator) -> dict:
    """Create a short narrative with slang and explicit misconduct cues."""

    kind, label = _BEHAVIOR_CASE_TYPES[int(rng.integers(0, len(_BEHAVIOR_CASE_TYPES)))]
    actor = rng.choice(_BEHAVIOR_ACTORS)
    context = rng.choice(_BEHAVIOR_CONTEXTS)
    flag_desc, keyword = _INAPPROPRIATE_FLAGS[int(rng.integers(0, len(_INAPPROPRIATE_FLAGS)))]
    reaction = rng.choice(_BEHAVIOR_REACTIONS)
    escalation = rng.choice(_BEHAVIOR_ESCALATION)
    slang = rng.choice(_MEXICAN_SLANG)

    title = f"{label} - {flag_desc.capitalize()}"
    label_lower = label.lower()
    body_template = rng.choice(
        [
            (
                "{actor} reportó que {context} se perciben señales de {keyword}; "
                "{slang}. Además, {reaction} y {escalation}."
            ),
            (
                "Testimonio de {actor}: {context} persiste {flag_desc}, {slang}. "
                "El equipo indicó que {reaction} y {escalation}."
            ),
            (
                "En seguimiento a {label_lower}, {actor} detalló que {context} hay {flag_desc}; "
                "{slang}. Se documentó que {reaction} y {escalation}."
            ),
        ]
    )
    body = body_template.format(
        actor=actor,
        context=context,
        flag_desc=flag_desc,
        keyword=keyword,
        slang=slang,
        reaction=reaction,
        escalation=escalation,
        label=label,
        label_lower=label_lower,
    )

    return {
        "behavior_case_type": kind,
        "behavior_case_title": title,
        "behavior_case_body": body,
    }


def generate_diverse_dataset(n_records: int = 6_000, seed: int | None = 42) -> pd.DataFrame:
    """Create a synthetic dataframe with 6k highly diverse transactions.

    Parameters
    ----------
    n_records:
        Número de transacciones a generar. Por defecto 6 000.
    seed:
        Semilla opcional para obtener resultados reproducibles.

    Returns
    -------
    pandas.DataFrame
        Dataset listo para ser consumido por :func:`coi_fraud.run_pipeline`.
    """

    if n_records < 1:
        raise ValueError("n_records debe ser mayor a 0")

    rng = np.random.default_rng(seed)
    persona_pool = max(240, max(1, n_records // 24))
    personas = _build_personas(rng, size=persona_pool)
    persona_ids = [p.user_id for p in personas]

    start = datetime(2020, 1, 1)
    end = datetime(2025, 12, 31)
    delta_days = (end - start).days

    rows = []
    for idx in range(n_records):
        sender = personas[rng.integers(0, len(personas))]
        receiver = personas[rng.integers(0, len(personas))]
        if receiver.user_id == sender.user_id and rng.random() < 0.85:
            receiver = personas[rng.integers(0, len(personas))]

        load_dt = start + timedelta(days=int(rng.integers(0, delta_days)))
        load_dt = load_dt + timedelta(minutes=int(rng.integers(0, 60 * 24)))
        amount = round(_scenario_amount(rng), 2)
        description = _build_description(rng)
        origin = rng.choice(_ORIGIN_APPS)

        teammates = _sample_teammates(
            rng,
            pool=persona_ids,
            size=rng.integers(1, min(6, len(persona_ids)))
        )
        managers = _sample_manager_chain(rng, pool=persona_ids)

        row = {
            "user_id": sender.user_id,
            "receptor-user_id": receiver.user_id,
            "load_date": load_dt.strftime("%Y-%m-%d %H:%M"),
            "movement_amount": amount,
            "transaction_desc": description,
            "origin_application_id": origin,
            "companeros_de_equipo": teammates,
        }

        for i, manager in enumerate(managers, start=1):
            row[f"manager_{i}_user_id"] = manager

        row.update(_sample_persona_metadata(sender, "envio"))
        row.update(_sample_persona_metadata(receiver, "receptor"))

        # Garantiza diversidad adicional en conceptos con metadatos complementarios
        row.update(
            {
                "tx_channel": rng.choice([
                    "transferencia",
                    "spei",
                    "pago_tarjeta",
                    "crypto",
                    "cash_pool",
                    "inversion",
                    "fondos_mutuos",
                ]),
                "tx_context": rng.choice([
                    "programa_inclusion",
                    "expansion_regional",
                    "laboratorio_ideas",
                    "operacion_contingencia",
                    "relocalizacion",
                    "experiencia_cliente",
                    "auditoria_etica",
                    "alianza_cientifica",
                ]),
                "tx_tags": ";".join(
                    rng.choice(_LANGUAGE_SNIPPETS + _DETAIL_TAGS, size=3, replace=False)
                ),
            }
        )

        row.update(_build_behavioral_case(rng))

        rows.append(row)

    df = pd.DataFrame(rows)

    manager_cols = [col for col in df.columns if col.startswith("manager_")]
    for col in manager_cols:
        df[col] = df[col].fillna(pd.NA)

    # Añade columnas opcionales para robustez en pruebas de ingestión
    df["receptor-tenure_years"] = df["receptor-gf_worker_hiring_date"].map(
        lambda d: round(((datetime.now() - datetime.strptime(d, "%Y-%m-%d")).days) / 365.25, 2)
    )

    return df


__all__ = ["generate_diverse_dataset"]

