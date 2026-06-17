# CLAUDE.md — Proyecto 2: ETL de gran escala enriquecido con GenAI

Memoria de trabajo del repo entre sesiones. Léelo antes de retomar.

## Qué es esto

Tercer repo de un portafolio de 3 (datos + IA) para defender en entrevistas. Demuestra
**aplicar IA de vanguardia sobre Big Data sin reventar el rendimiento**: un ETL distribuido
en PySpark que extrae texto no estructurado a gran escala, lo estructura/categoriza con
modelos preentrenados de **Hugging Face** dentro de **UDFs de Spark optimizadas**, y lo
persiste en un **lakehouse Delta Lake**. El núcleo se construye **una vez** y se demuestra
sobre tres dominios cambiando solo `config` + esquema (salud → educación → banca), no
reescribiendo el pipeline.

**Hilo conductor (ventaja estadística del autor):** confianza por extracción (score del
modelo), umbral/calibración **honesta** (no fingir probabilidades calibradas sin
ground-truth), métricas de calidad sobre la salida estructurada y **drift** de la
distribución del texto/etiquetas.

## Stack y por qué

| Pieza | Elección | Por qué |
|---|---|---|
| Gestor de entorno | **uv** | Rápido, lockfile reproducible, no toca el Python del sistema |
| Procesamiento | **PySpark 4.x** | ETL distribuido; mismo código en `local[*]` y EMR/Dataproc |
| Modelos | **Hugging Face** (transformers) | NER y sentimiento preentrenados, en español/multilingüe |
| Inferencia a escala | **pandas UDF / `mapInPandas`** + singleton por executor | Lotes vectorizados Arrow + modelo cargado una vez por proceso = sin cuello de botella |
| torch | **ruedas CPU** | Evita descargar el build CUDA en laptop/CI |
| Validación | **Pandera** | Contrato explícito de la **salida** estructurada |
| Lakehouse | **Delta Lake** (`delta-spark`) | Cero fricción local con PySpark, ACID, time travel, schema evolution |
| Confianza/drift | **scipy** (KS/PSI) | Sello estadístico sobre la salida |
| Backend swap | env var `GENAI_BACKEND` (`mock\|hf\|auto`) | Tests/CI sin descarga ni GPU (espejo del patrón de P3) |

## Estructura

- `src/genai_etl/` — núcleo agnóstico al dominio:
  - `config.py` (Pydantic + `load_config()`), `spark.py` (JDK-21 discovery + Delta).
  - `ingest/` (lectura de texto crudo), `models/` (backends HF + Mock, registry singleton,
    inferencia vectorizada), `schemas/` (contrato Pandera de salida), `validate/`,
    `lakehouse/` (escritura Delta particionada), `quality/` (confianza + drift),
    `enrich/` (orquestación del pipeline).
- `configs/*.yaml` — un archivo por dominio.
- `tests/` — pytest por componente; `conftest.py` fija `GENAI_BACKEND=mock`.
- `scripts/` — `generate_synthetic.py`, `download_data.py`, `run_pipeline.py`.
- `iac/` — aprovisionamiento EMR/Dataproc documentado + `local-spark-submit.sh`.
- `docs/` — documentación en 3 capas + `spark-genai-udf.md` (el crux) y `podman-vs-docker.md`.

## El crux — HF en UDFs de Spark sin cuellos de botella

1. **`mapInPandas` / pandas UDF (Arrow)** en vez de UDF fila-a-fila → transferencia
   columnar en bloque; el modelo recibe **lotes**, no filas.
2. **Singleton del modelo por executor** (`models/registry.py`, cache module-level lazy):
   cada proceso worker carga los pesos **una vez**, no por fila/partición. Solo viaja el
   `model_name` (str); **no** se hace broadcast del objeto modelo.
3. **Inferencia batched** (`batch_size`, `truncation`, `max_length`).
4. **`torch.set_num_threads(1)`** por worker (Spark paraleliza por particiones; evita
   sobre-suscripción de CPU).
5. Mismo código en laptop y cluster cambiando solo el master (`SPARK_MASTER`).

## Estrategia de modelos (Opus para pensar, Sonnet para ejecutar)

- **Opus 4.8** (`claude-opus-4-8`): diseño, arquitectura, trade-offs, redacción del
  documento técnico + diagramas. El diseño de este proyecto ya está cerrado y aprobado.
- **Sonnet 4.6** (`claude-sonnet-4-6`): ejecutar lo decidido — código, tests, docstrings,
  referencia de código, refactors menores.
- Antes de cada bloque grande, decir en qué modelo conviene estar y por qué.
- Nota: este proyecto se está ejecutando en Opus 4.8 por elección explícita de Felipe.

## Entorno (Fedora 44)

- **JDK 21 portátil** en `~/.local/share/jvm/jdk-21*` (el Java del sistema es 25,
  incompatible con Spark 4.x). `spark.get_spark()` lo descubre y fija `JAVA_HOME` solo
  para el proceso de Spark. NUNCA instalar a nivel de sistema sin preguntar.
- **uv** para todo; nunca pip/venv del sistema.

## Comandos

```bash
uv sync
uv run pre-commit install
GENAI_BACKEND=mock uv run python scripts/generate_synthetic.py --domain health --out data/raw/health.csv
GENAI_BACKEND=mock uv run python scripts/run_pipeline.py --config configs/health.yaml
GENAI_BACKEND=mock uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Convenciones

- Plan-first, incremental, tests verdes en cada fase.
- **Conventional Commits** en español (`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
- Docs/comentarios/docstrings en español; identificadores de código en inglés.
- Multi-dominio obligatorio: salud + educación + banca con el mismo núcleo.
- Documentación obligatoria en 3 capas (técnico, referencia de código, glosario).

## Bitácora de decisiones

- **2026-06-17** — Lakehouse **Delta Lake** (no Iceberg): menor fricción local con
  PySpark, mismo discurso ACID/time travel; Iceberg pediría catálogo externo.
- **2026-06-17** — Datos y modelos HF en **español** (multilingües ligeros): conecta con
  la experiencia ISP del autor y el mercado local.
- **2026-06-17** — **Sí** contenedor spark-submit (Podman rootless), por consistencia con P1.
- **2026-06-17** — `pyspark` y `delta-spark` fijados a `>=4.0,<4.1` (acoplamiento estricto
  entre ambos). torch desde el índice de ruedas CPU de PyTorch.
- **2026-06-17** — Backend swap por `GENAI_BACKEND` con `MockBackend` determinista, espejo
  del patrón `HEALER_BACKEND` de P3, para tests/CI sin red ni descarga.

## Estado actual

**Proyecto completo ✅** — 12 commits, 48 tests verdes (mock), contenedor Podman verificado.

- Fase 0 — Scaffold (uv, ruff/pytest, CI con Java, pre-commit) ✅
- Fase 1 — Config Pydantic multi-dominio (3 YAML) ✅
- Fase 2 — SparkSession con JDK-21 discovery + Delta ✅
- Fase 3 — Backends HF + MockBackend + registry singleton ✅
- Fase 4 — Inferencia vectorizada `mapInPandas` (el crux) ✅
- Fase 5 — Esquemas Pandera de salida + validación ✅
- Fase 6 — Lakehouse Delta particionado + time travel ✅
- Fase 7 — Confianza/calibración (ECE) + drift (PSI/KS) ✅
- Fase 8 — Pipeline E2E + generador sintético + entrypoint ✅
- Fase 9 — IaC documentada (EMR/Dataproc + local-spark-submit) ✅
- Fase 10 — Documentación 3 capas + write-up del crux ✅
- Fase 11 — Contenedor spark-submit (Podman rootless, JDK 21) ✅

**Nota de entorno:** `python:3.12-slim` ahora es Debian trixie; el contenedor usa
`openjdk-21-jre-headless` (no 17). El contenedor corrió end-to-end escribiendo la tabla
Delta en el volumen `./lakehouse` montado.
