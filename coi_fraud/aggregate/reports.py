
from .transactions import build_tx_table
from .pairs import build_pair_monthly
from .persons import build_person_monthly
from .concepts import build_concept_tables
def build_all_reports(df):
    tx = build_tx_table(df)
    pairs = build_pair_monthly(df)
    persons = build_person_monthly(df)
    agg_concepto, agg_persona_concepto, agg_par_concepto = build_concept_tables(df)
    return {
        "tx_transacciones_priorizadas": tx,
        "agg_par_mensual": pairs,
        "agg_persona_mensual": persons,
        "agg_concepto_mensual": agg_concepto,
        "agg_persona_concepto_mensual": agg_persona_concepto,
        "agg_par_concepto_mensual": agg_par_concepto,
    }
