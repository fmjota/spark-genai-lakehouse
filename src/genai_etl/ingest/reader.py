"""Lectura de texto crudo (CSV/Parquet) a un DataFrame de Spark normalizado.

Normaliza la entrada de cada dominio a un contrato mínimo: columnas `id` (str) y `text`
(str). Así el resto del pipeline (UDF de inferencia, validación, escritura Delta) es
agnóstico al nombre original de las columnas en cada fuente.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from genai_etl.config import DomainConfig


def read_text(spark: SparkSession, cfg: DomainConfig) -> DataFrame:
    """Lee la fuente cruda del dominio y la normaliza a columnas `id` y `text`.

    Args:
        spark: SparkSession activa.
        cfg: config del dominio (rutas y nombres de columnas).

    Returns:
        DataFrame con columnas `id` (str) y `text` (str). Si el config no define
        `id_column`, se genera un id determinista por fila.

    Side effects:
        Lee del sistema de archivos (CSV con header o Parquet, según extensión).
    """
    path = cfg.data.raw_path
    reader = spark.read
    if path.endswith(".parquet"):
        df = reader.parquet(path)
    else:
        df = reader.option("header", True).option("multiLine", True).csv(path)

    text_col = cfg.data.text_column
    if text_col not in df.columns:
        raise ValueError(
            f"La columna de texto '{text_col}' no existe en {path}. Columnas: {df.columns}"
        )

    if cfg.data.id_column and cfg.data.id_column in df.columns:
        id_expr = F.col(cfg.data.id_column).cast("string")
    else:
        # id determinista por orden de llegada (monotónico, estable dentro de la corrida).
        id_expr = F.monotonically_increasing_id().cast("string")

    return df.select(
        id_expr.alias("id"),
        F.col(text_col).cast("string").alias("text"),
    )
