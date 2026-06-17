"""Inferencia HF vectorizada sobre Spark — el núcleo anti-cuello-de-botella.

Aplica un backend de NLP a una columna de texto usando `DataFrame.mapInPandas`, que mueve
los datos entre la JVM y Python en bloques **Arrow** (columnar) y entrega a la función
`pandas.DataFrame` por partición. Dentro de la función:

1. El backend se obtiene del **registry singleton** (`get_backend`), de modo que el modelo
   se carga **una vez por proceso worker**, no por fila ni por partición.
2. La inferencia corre **por lotes** (`batch_size`, `max_length`, `truncation`), no
   fila a fila.
3. Si el backend es real (HF/torch), se limita a **1 hilo por worker** para que el
   paralelismo lo dé Spark (por particiones) y no haya sobre-suscripción de CPU.

El mismo código corre en `local[*]` y en un cluster EMR/Dataproc cambiando solo el master.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

from genai_etl.config import DomainConfig

_ENTITY_STRUCT = StructType(
    [
        StructField("text", StringType()),
        StructField("type", StringType()),
        StructField("start", IntegerType()),
        StructField("end", IntegerType()),
        StructField("score", DoubleType()),
    ]
)


def output_schema(task: str) -> StructType:
    """Esquema Spark de salida según la tarea (sentimiento o NER)."""
    base = [StructField("id", StringType()), StructField("text", StringType())]
    if task == "sentiment":
        return StructType(
            [*base, StructField("label", StringType()), StructField("confidence", DoubleType())]
        )
    return StructType(
        [
            *base,
            StructField("entities", ArrayType(_ENTITY_STRUCT)),
            StructField("n_entities", IntegerType()),
            StructField("confidence", DoubleType()),
        ]
    )


def _make_mapper(task, model_name, batch_size, max_length, mode):
    """Crea la función de `mapInPandas` cerrando solo sobre primitivos serializables."""

    def _mapper(itr: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        # Importación dentro del worker: el modelo se cachea por proceso (singleton).
        from genai_etl.models.registry import get_backend

        backend = get_backend(task, model_name, mode)

        # Backends reales: 1 hilo por worker (Spark ya paraleliza por particiones).
        if not backend.__class__.__name__.startswith("Mock"):
            try:
                import torch

                torch.set_num_threads(1)
            except Exception:
                pass

        for pdf in itr:
            texts = pdf["text"].fillna("").astype(str).tolist()
            preds = backend.infer(texts, batch_size=batch_size, max_length=max_length)
            if task == "sentiment":
                yield pd.DataFrame(
                    {
                        "id": pdf["id"].to_numpy(),
                        "text": texts,
                        "label": [p["label"] for p in preds],
                        "confidence": [float(p["confidence"]) for p in preds],
                    }
                )
            else:
                ents = [p["entities"] for p in preds]
                yield pd.DataFrame(
                    {
                        "id": pdf["id"].to_numpy(),
                        "text": texts,
                        "entities": ents,
                        "n_entities": [len(e) for e in ents],
                        "confidence": [float(p["confidence"]) for p in preds],
                    }
                )

    return _mapper


def enrich_with_nlp(df: DataFrame, cfg: DomainConfig, mode: str | None = None) -> DataFrame:
    """Estructura la columna `text` con el modelo de NLP del dominio, en paralelo.

    Args:
        df: DataFrame con columnas `id` y `text` (salida de `ingest.reader.read_text`).
        cfg: config del dominio (tarea, modelo, batch_size, max_length).
        mode: fuerza el backend (`mock`|`hf`|`auto`); si es None se lee de `GENAI_BACKEND`.

    Returns:
        DataFrame estructurado. Sentimiento: `id, text, label, confidence`.
        NER: `id, text, entities (array<struct>), n_entities, confidence`.
    """
    mapper = _make_mapper(
        cfg.model.task,
        cfg.model.model_name,
        cfg.model.batch_size,
        cfg.model.max_length,
        mode,
    )
    return df.mapInPandas(mapper, schema=output_schema(cfg.model.task))
