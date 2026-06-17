"""Contrato Pandera de la salida de NER.

Valida la tabla estructurada de entidades: número de entidades no negativo, confianza en
[0, 1] y —si el dominio declara `entity_types`— que todos los tipos extraídos pertenezcan
al conjunto esperado. La columna `entities` es una lista de dicts por fila.
"""

from __future__ import annotations

from pandera.pandas import Check, Column, DataFrameSchema


def ner_schema(entity_types: list[str] | None = None) -> DataFrameSchema:
    """Esquema de la salida de NER; valida los tipos contra `entity_types` si se entregan."""
    checks = []
    if entity_types:
        allowed = set(entity_types)

        def _tipos_validos(df) -> bool:
            return all(e["type"] in allowed for ents in df["entities"] for e in ents)

        checks.append(Check(_tipos_validos, error=f"tipo de entidad fuera de {sorted(allowed)}"))

    return DataFrameSchema(
        columns={
            "id": Column(str, nullable=False),
            "text": Column(str, nullable=False),
            "entities": Column(object, nullable=False),
            "n_entities": Column(int, Check.ge(0), nullable=False),
            "confidence": Column(float, Check.in_range(0.0, 1.0), nullable=False),
        },
        checks=checks,
        strict=False,
        coerce=True,
        name="ner_output",
    )
