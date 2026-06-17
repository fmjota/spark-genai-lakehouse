"""Ejecuta la validación Pandera sobre la salida estructurada (en pandas).

La inferencia produce un DataFrame de Spark; para validar el contrato lo materializamos a
pandas (el volumen de salida de un demo es modesto y la capa de calidad también opera en
pandas). `lazy=True` acumula todos los errores en vez de fallar en el primero.
"""

from __future__ import annotations

import pandas as pd

from genai_etl.config import DomainConfig
from genai_etl.schemas import get_output_schema


def validate_output(pdf: pd.DataFrame, cfg: DomainConfig) -> pd.DataFrame:
    """Valida `pdf` contra el esquema del dominio.

    Args:
        pdf: salida estructurada en pandas.
        cfg: config del dominio.

    Returns:
        El mismo DataFrame (con tipos coercionados) si la validación pasa.

    Raises:
        pandera.errors.SchemaErrors: si alguna fila/columna viola el contrato.
    """
    schema = get_output_schema(cfg)
    return schema.validate(pdf, lazy=True)
