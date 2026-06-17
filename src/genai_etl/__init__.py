"""genai_etl — ETL distribuido enriquecido con GenAI.

Núcleo agnóstico al dominio: ingesta de texto no estructurado en PySpark, estructuración
con modelos de Hugging Face dentro de UDFs vectorizadas, validación de la salida con
Pandera, métricas de confianza/drift y persistencia en un lakehouse Delta. Se parametriza
por `configs/<dominio>.yaml` (salud / educación / banca) sin reescribir el pipeline.
"""

__version__ = "0.1.0"
