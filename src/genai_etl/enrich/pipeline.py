"""Pipeline end-to-end multi-dominio.

Encadena las etapas del núcleo, todas agnósticas al dominio (se parametrizan por el config):

    ingesta de texto  →  inferencia HF vectorizada (Spark)  →  validación Pandera
                      →  métricas de confianza  →  escritura en lakehouse Delta

Devuelve un resumen con la ruta de la tabla Delta y las métricas de confianza/calidad.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from genai_etl.config import DomainConfig
from genai_etl.ingest.reader import read_text
from genai_etl.lakehouse.delta_writer import write_delta
from genai_etl.models.inference import enrich_with_nlp
from genai_etl.quality.confidence import confidence_summary, flag_low_confidence
from genai_etl.validate.runner import validate_output


def run_pipeline(
    cfg: DomainConfig,
    spark: SparkSession | None = None,
    mode: str | None = None,
    ingest_date: str | None = None,
) -> dict:
    """Ejecuta el pipeline completo para un dominio.

    Args:
        cfg: config del dominio.
        spark: SparkSession; si es None se crea una local (con Delta).
        mode: backend a usar (`mock`|`hf`|`auto`); si es None se lee de `GENAI_BACKEND`.
        ingest_date: fecha de ingesta `YYYY-MM-DD` para el particionado.

    Returns:
        Resumen: dominio, tarea, ruta Delta y métricas de confianza/calidad.

    Side effects:
        Escribe la salida estructurada como tabla Delta en `cfg.lakehouse.path`.
    """
    if spark is None:
        from genai_etl.spark import get_spark

        spark = get_spark()

    raw = read_text(spark, cfg)
    structured = enrich_with_nlp(raw, cfg, mode=mode)

    # Validación + métricas en pandas (el contrato y la capa estadística operan ahí).
    pdf = structured.toPandas()
    validate_output(pdf, cfg)
    pdf = flag_low_confidence(pdf, cfg.quality.min_confidence)
    summary = confidence_summary(pdf, cfg.quality.min_confidence)

    path = write_delta(structured, cfg, ingest_date=ingest_date)

    return {
        "domain": cfg.domain,
        "task": cfg.model.task,
        "delta_path": path,
        **summary,
    }
