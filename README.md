# Proyecto 2 — ETL de gran escala enriquecido con GenAI

> ⏳ En construcción. Documentación completa (pitch, diagrama, resultados) en la fase 10.

Pipeline de ETL **distribuido** (PySpark) que extrae texto no estructurado a gran escala y
lo **estructura/categoriza** con modelos preentrenados de Hugging Face dentro de **UDFs de
Spark optimizadas** (sin cuellos de botella), persistiendo en un **lakehouse Delta Lake**.
El mismo núcleo se reutiliza sobre tres dominios cambiando solo la config:

| Dominio | Caso | Técnica |
|---|---|---|
| Salud | Estructurar notas clínicas / eventos adversos | NER de fármacos y síntomas |
| Educación | Evaluaciones docentes y feedback a escala | Sentimiento + tópicos |
| Banca/retail | Reseñas de productos / noticias financieras | Sentimiento + entidades |

**Sello estadístico:** confianza por extracción, umbral/calibración honesta, métricas de
calidad sobre la salida y drift de distribución de texto.

## Cómo correr (vista previa)

```bash
uv sync
GENAI_BACKEND=mock uv run python scripts/generate_synthetic.py --domain health --out data/raw/health.csv
GENAI_BACKEND=mock uv run python scripts/run_pipeline.py --config configs/health.yaml
```

## Documentación

- `docs/vision-tecnica.md` — cómo y por qué funciona por dentro.
- `docs/spark-genai-udf.md` — el patrón HF-en-Spark sin cuellos de botella.
- `docs/referencia-codigo.md` — archivo por archivo.
- `docs/glosario.md` — términos.
- `CLAUDE.md` — memoria de trabajo y decisiones.
