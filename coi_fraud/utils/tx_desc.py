import re
import pandas as pd

__all__ = ["split_transaction_desc", "_split_transaction_desc_series"]


def _split_transaction_desc_series(series: pd.Series) -> pd.DataFrame:
    """
    Divide descripciones heterogéneas de transacciones en partes estándar.

    Entrada:
        series: pandas.Series con textos de la columna 'transaction_desc'.

    Salida (DataFrame):
        - original: cadena original
        - service: texto antes del primer identificador numérico (p.ej., 'BNET', 'PREST.', 'CARGO METAS', 'Trp redondeo de tarjeta')
        - count: entero inicial si el texto empieza con un número (p.ej., '8 Trp ...')
        - id1: primer token con dígitos (o dígitos mezclados con X), p.ej. '299XXX4746'
        - id2: segundo token con dígitos, si existe (p.ej., '202509XXX')
        - detail: texto entre id1 e id2; si no hay id2, texto posterior a id1
        - code: sufijo tipo código operativo al final (p.ej., 'K82','K83','N06','H09','KN06') incluso pegado
    """
    if not isinstance(series, pd.Series):
        raise TypeError("_split_transaction_desc_series espera un pandas.Series")

    code_re = re.compile(r'([A-Z]{1,2}\d{2})\s*$', re.ASCII)
    nbsp_re = re.compile(r'\u00A0')
    letters = r"A-Za-zÁÉÍÓÚÜÑáéíóúüñ\."
    # Solo separa cuando hay una palabra de ≥3 letras (no precedida por dígito) seguida por dígito
    # Evita romper tokens como 980XXX4481
    pre_num_re = re.compile(rf'(?<!\d)([{letters}]{{3,}})(\d)')
    collapse_ws = re.compile(r'\s+')
    token_with_digit = re.compile(r'\b\S*\d\S*\b')

    def parse_one(raw):
        if pd.isna(raw):
            return {
                'original': raw,
                'service': None,
                'count': None,
                'id1': None,
                'id2': None,
                'detail': None,
                'code': None,
            }

        t = str(raw)
        t = nbsp_re.sub(' ', t)
        t = collapse_ws.sub(' ', t).strip()

        # 1) Código al final (soporta pegado, p.ej., ...000H09)
        code = None
        m = code_re.search(t)
        if m:
            code = m.group(1)
            t = t[:m.start()].rstrip()

        # 2) Separar letras<->dígitos para palabras largas: 'METAS2323' -> 'METAS 2323', 'tarjeta11...' -> 'tarjeta 11...'
        t = pre_num_re.sub(r'\1 \2', t)
        t = collapse_ws.sub(' ', t).strip()

        # 3) Contador inicial (p.ej., '8 Trp ...')
        count = None
        mcount = re.match(r'^(\d+)\b\s+', t)
        if mcount:
            try:
                count = int(mcount.group(1))
            except Exception:
                count = None
            t = t[mcount.end():].strip()

        # 4) Identificadores con dígitos (conserva 'XXX' intercaladas)
        digit_tokens = token_with_digit.findall(t)
        id1 = digit_tokens[0] if digit_tokens else None
        id2 = digit_tokens[1] if len(digit_tokens) > 1 else None

        # 5) service / detail
        if id1:
            pos1 = t.find(id1)
            pre = t[:pos1].strip()
            post = t[pos1 + len(id1):].strip()
            if id2:
                pos2 = post.find(id2)
                if pos2 >= 0:
                    detail = post[:pos2].strip() or None
                else:
                    detail = post or None
            else:
                detail = post or None
            service = pre or None
        else:
            service = t or None
            detail = None

        return {
            'original': raw,
            'service': service,
            'count': count,
            'id1': id1,
            'id2': id2,
            'detail': detail,
            'code': code,
        }

    return series.apply(parse_one).apply(pd.Series)


def split_transaction_desc(
    df: pd.DataFrame,
    col: str = "transaction_desc",
    *,
    prefix: str = "tx_",
    keep_original: bool = True,
) -> pd.DataFrame:
    """
    Añade las columnas parseadas al DataFrame a partir de 'col'.

    Parámetros:
        df: DataFrame de entrada
        col: nombre de la columna con la descripción de la transacción
        prefix: prefijo para las nuevas columnas (p.ej., 'tx_')
        keep_original: si False, elimina la columna original

    Retorna:
        DataFrame con columnas agregadas:
        f'{prefix}service', f'{prefix}count', f'{prefix}id1',
        f'{prefix}id2', f'{prefix}detail', f'{prefix}code'
    """
    parts = _split_transaction_desc_series(df[col])
    # Renombrar y seleccionar columnas útiles (omitimos 'original')
    parts = parts[['service', 'count', 'id1', 'id2', 'detail', 'code']].add_prefix(prefix)
    out = df.copy()
    out = pd.concat([out, parts], axis=1)
    if not keep_original:
        out = out.drop(columns=[col])
    return out
