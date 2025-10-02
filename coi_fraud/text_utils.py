"""Utilidades de texto compartidas para normalizar conceptos crudos."""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable

import pandas as pd

# Patrón de separación reutilizado al limpiar conceptos crudos.
_CONCEPT_SPLIT_PATTERN = re.compile(r"[\s,;|/]+")

# Codigos y expresiones que suelen acompañar referencias BNET u otros folios.
_NOISE_TOKENS = {
    "BNET",
    "B-NET",
    "B_NET",
    "BN",
    "BNTA",
    "PAY",
}
_NOISE_PATTERN = re.compile(r"^(?:N\d{2,}|ID\d{2,}|FOLIO\d{2,}|REF\d{2,}|OP\d{2,}|TRX\d{2,})$")


def clean_raw_concept(value: Any) -> str:
    """Limpia un concepto libre eliminando prefijos/códigos comunes.

    Ante cualquier error se devuelve el texto original para garantizar
    robustez, cumpliendo el requerimiento de resiliencia a fallos.
    """

    original = "" if value is None else str(value)
    try:
        text = original.strip()
        if not text or text.lower() == "nan":
            return ""
        cleaned_tokens: list[str] = []
        for token in _CONCEPT_SPLIT_PATTERN.split(text):
            token = token.strip()
            if not token:
                continue
            normalized = re.sub(r"[^A-Z0-9]", "", token.upper())
            if not normalized:
                continue
            if normalized in _NOISE_TOKENS:
                continue
            if _NOISE_PATTERN.fullmatch(normalized):
                continue
            cleaned_tokens.append(token)
        cleaned = " ".join(cleaned_tokens).strip()
        return cleaned if cleaned else original
    except Exception:
        return original


def normalize_clean_concept(value: Any) -> str:
    """Normaliza un concepto crudo limpiando ruido y homogenizando el texto."""

    cleaned = clean_raw_concept(value)
    if not cleaned:
        return ""
    text = unicodedata.normalize("NFKD", cleaned)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text


def first_non_empty_series(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    """Devuelve la primera columna no vacía por fila como serie de strings."""

    if df.empty:
        return pd.Series([], index=df.index, dtype="string")
    result = pd.Series([""] * len(df), index=df.index, dtype="string")
    for col in columns:
        if col not in df:
            continue
        candidate = df[col].fillna("").astype(str).str.strip()
        candidate = candidate.mask(candidate.str.fullmatch(r"(?i)nan"), "")
        mask = result.str.len() == 0
        if mask.any():
            result.loc[mask] = candidate.loc[mask]
    return result
