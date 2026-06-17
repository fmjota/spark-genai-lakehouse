"""Esquemas Pandera de la salida estructurada, despachados por tarea.

El contrato de salida depende de la **tarea** (sentimiento vs NER), no del dominio: por eso
se organiza por tarea y `get_output_schema` elige el esquema según `cfg.model.task`,
parametrizándolo con las etiquetas / tipos de entidad del dominio.
"""

from __future__ import annotations

from pandera.pandas import DataFrameSchema

from genai_etl.config import DomainConfig
from genai_etl.schemas.ner import ner_schema
from genai_etl.schemas.sentiment import sentiment_schema


def get_output_schema(cfg: DomainConfig) -> DataFrameSchema:
    """Devuelve el esquema Pandera de la salida del dominio según su tarea."""
    if cfg.model.task == "sentiment":
        return sentiment_schema(cfg.model.labels)
    return ner_schema(cfg.model.entity_types)


__all__ = ["get_output_schema", "ner_schema", "sentiment_schema"]
