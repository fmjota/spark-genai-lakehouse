"""Contrato Pandera de la salida de sentimiento.

Valida la tabla estructurada que produce la inferencia: etiqueta dentro del conjunto
permitido y confianza en [0, 1]. Es la misma filosofía de P1/P3 (fallar temprano con un
contrato explícito), pero aplicada a la **salida** del modelo, que es donde está el valor.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema


def sentiment_schema(labels: list[str]) -> DataFrameSchema:
    """Esquema de la salida de sentimiento para un conjunto de etiquetas dado."""
    return DataFrameSchema(
        columns={
            "id": Column(str, nullable=False),
            "text": Column(str, nullable=False),
            "label": Column(str, Check.isin(labels), nullable=False),
            "confidence": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        },
        strict=False,
        coerce=True,
        name="sentiment_output",
    )
