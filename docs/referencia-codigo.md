# Referencia de código (archivo por archivo)

Complementa los docstrings de cada módulo. Mantener sincronizado con el código.

## `src/genai_etl/`

| Archivo | Propósito | Inputs | Outputs | Depende de |
|---|---|---|---|---|
| `config.py` | Esquema Pydantic del config de dominio y `load_config()` | ruta YAML | `DomainConfig` | pydantic, pyyaml |
| `spark.py` | Fábrica de SparkSession con JDK 17/21 + extensión Delta + Arrow | — | `SparkSession` | pyspark, delta |
| `ingest/reader.py` | Lee texto crudo y lo normaliza a `id, text` | `SparkSession`, `DomainConfig` | DataFrame Spark | pyspark |
| `models/backends.py` | Backends `mock`/`hf` (sentimiento, NER) + selección por env | `texts` | `list[dict]` por texto | transformers (lazy) |
| `models/registry.py` | Cache singleton del backend (una carga por executor) | task, model_name | `HFBackend` | backends |
| `models/inference.py` | UDF vectorizada `mapInPandas` + esquema de salida | DataFrame `id,text`, `DomainConfig` | DataFrame estructurado | pyspark, pandas, registry |
| `schemas/sentiment.py` | Contrato Pandera de salida de sentimiento | `labels` | `DataFrameSchema` | pandera |
| `schemas/ner.py` | Contrato Pandera de salida de NER | `entity_types` | `DataFrameSchema` | pandera |
| `schemas/__init__.py` | `get_output_schema(cfg)` despacha por tarea | `DomainConfig` | `DataFrameSchema` | sentiment, ner |
| `validate/runner.py` | Valida la salida (pandas) contra su contrato | `pdf`, `DomainConfig` | `pdf` validado / `SchemaErrors` | schemas |
| `lakehouse/delta_writer.py` | Escritura Delta particionada + time travel | `df`, `DomainConfig` | ruta / DataFrame | pyspark, delta |
| `quality/confidence.py` | Resumen de confianza, fiabilidad y ECE | `pdf`, `min_confidence` | dict / DataFrame / float | numpy, pandas |
| `quality/drift.py` | PSI (num/categórico), KS, reporte de drift | `ref`, `new`, `cfg` | dict | numpy, scipy |
| `enrich/pipeline.py` | Orquesta ingesta→UDF→validación→calidad→Delta | `DomainConfig` | resumen dict | todos los anteriores |

## `configs/`

| Archivo | Dominio | Tarea | Modelo HF |
|---|---|---|---|
| `health.yaml` | salud | ner | `lcampillos/roberta-es-clinical-trials-ner` |
| `education.yaml` | educación | sentiment | `pysentimiento/robertuito-sentiment-analysis` |
| `banking.yaml` | banca/retail | sentiment | `pysentimiento/robertuito-sentiment-analysis` |

## `scripts/`

| Archivo | Propósito |
|---|---|
| `generate_synthetic.py` | Genera texto sintético en español por dominio (con `etiqueta_real` para calibración) |
| `download_data.py` | Descarga opcional de datasets públicos en español (vía HF `datasets`) |
| `run_pipeline.py` | Entrypoint CLI: corre un dominio y resume en consola (rich) |

## `iac/`

| Archivo | Propósito |
|---|---|
| `local-spark-submit.sh` | Equivalente local del job (spark-submit + Delta + JDK 21) |
| `emr-cluster.json` | Plantilla de cluster EMR efímero (documental) |
| `dataproc-cluster.sh` | Creación/envío/borrado de cluster Dataproc (DRY_RUN por defecto) |

## `tests/`

`conftest.py` fija `GENAI_BACKEND=mock` y expone fixtures `spark` (con skip si no hay JDK) y
`health_cfg`/`education_cfg`/`banking_cfg`. Un archivo `test_*.py` por componente.
