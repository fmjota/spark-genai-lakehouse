"""Carga y validación del config de dominio.

Cada dominio (salud, educación, banca) se describe con un YAML. El núcleo del pipeline es
agnóstico al dominio: lee este config para saber qué columna de texto procesar, qué tarea
de NLP correr (NER o sentimiento), con qué modelo de Hugging Face, dónde y cómo particionar
la tabla Delta, y qué umbrales de confianza/drift aplicar. Validar el config con Pydantic
es la primera línea de "fallar temprano": si el YAML está mal, fallamos antes de tocar
datos ni cargar modelos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# Tareas de NLP soportadas por el núcleo. El backend HF y el esquema de salida se eligen
# según este valor (sentimiento → etiqueta+score; NER → lista de entidades con span).
Task = Literal["sentiment", "ner"]


class DataPaths(BaseModel):
    raw_path: str
    # Columna que contiene el texto no estructurado a procesar.
    text_column: str
    # Columna identificadora (opcional); si falta se genera un id por fila.
    id_column: str | None = None


class ModelConfig(BaseModel):
    task: Task
    # Modelo de Hugging Face para el backend real (`GENAI_BACKEND=hf`). En `mock` se ignora.
    model_name: str
    # Tamaño de lote para la inferencia vectorizada dentro de cada partición de Spark.
    batch_size: int = Field(default=16, gt=0)
    # Truncado de secuencias: acota memoria y estabiliza el throughput.
    max_length: int = Field(default=256, gt=0)
    # Para sentimiento: etiquetas esperadas (validación/orden). Para NER: vacío.
    labels: list[str] = Field(default_factory=list)
    # Para NER: tipos de entidad esperados (ej. FARMACO, SINTOMA). Para sentimiento: vacío.
    entity_types: list[str] = Field(default_factory=list)


class LakehouseConfig(BaseModel):
    # Ruta de la tabla Delta de salida.
    path: str
    # Particionado del lakehouse: por dominio y fecha de ingesta por defecto.
    partition_by: list[str] = Field(default_factory=lambda: ["domain", "ingest_date"])
    # Modo de escritura Delta: append (acumular corridas) u overwrite.
    mode: Literal["append", "overwrite"] = "append"


class QualityConfig(BaseModel):
    # Umbral de confianza: por debajo se marca la extracción como de baja confianza.
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Umbrales de drift sobre features de texto/etiquetas (reutiliza la idea de P1).
    psi_threshold: float = Field(default=0.2, gt=0.0)
    ks_pvalue_threshold: float = Field(default=0.05, gt=0.0, le=1.0)


class DomainConfig(BaseModel):
    domain: str
    description: str = ""
    data: DataPaths
    model: ModelConfig
    lakehouse: LakehouseConfig
    quality: QualityConfig = Field(default_factory=QualityConfig)


def load_config(path: str | Path) -> DomainConfig:
    """Lee un YAML de dominio y lo valida contra el esquema Pydantic.

    Args:
        path: ruta al archivo YAML del dominio (ej. `configs/health.yaml`).

    Returns:
        DomainConfig: config validado y tipado.

    Raises:
        pydantic.ValidationError: si el YAML no cumple el esquema.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return DomainConfig.model_validate(raw)
