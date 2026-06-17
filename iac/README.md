# Aprovisionamiento de infraestructura (IaC)

El pipeline es **el mismo código** en la laptop y en un cluster: solo cambia el *master* de
Spark y el aprovisionamiento. Esto evita reescrituras al escalar y es el argumento central
del proyecto ("Big Data + GenAI sin reventar el rendimiento").

| Entorno | Archivo | Master | Cuándo |
|---|---|---|---|
| Local (Fedora) | `local-spark-submit.sh` | `local[*]` | desarrollo, demo, tests |
| AWS EMR | `emr-cluster.json` | YARN | volúmenes grandes en AWS |
| GCP Dataproc | `dataproc-cluster.sh` | YARN | volúmenes grandes en GCP |

## Local (ejecutable)

```bash
GENAI_BACKEND=mock iac/local-spark-submit.sh configs/health.yaml
```

Descubre un JDK 21 portátil (Spark 4.x no soporta Java 25), añade los paquetes de Delta y
lanza `scripts/run_pipeline.py`. Es el equivalente exacto de lo que corre en el cluster.

> La ruta más simple en local es `uv run python scripts/run_pipeline.py --config ...`:
> `get_spark()` ya configura Delta y el JDK. El script `spark-submit` se incluye para
> mostrar la forma idéntica a la del cluster.

## AWS EMR (documental, sin gasto)

```bash
aws emr create-cluster --cli-input-json file://iac/emr-cluster.json
```

Cluster **efímero** (`KeepJobFlowAliveWhenNoSteps: false` + `AutoTerminationPolicy`): se crea,
corre el step y se apaga; se paga solo el tiempo del job. EMR 7.x trae Spark y Java 17; el
bootstrap instala las dependencias Python en los nodos.

## GCP Dataproc (documental, sin gasto)

```bash
# DRY_RUN=1 por defecto: imprime los comandos sin ejecutarlos.
DRY_RUN=0 PROJECT=mi-proyecto BUCKET=mi-bucket iac/dataproc-cluster.sh
```

Crea un cluster con `--max-idle`, envía el job PySpark con las propiedades de Delta y borra
el cluster al terminar.

## Notas de costo y rendimiento

- **Clusters efímeros** + auto-apagado: nunca se paga un cluster ocioso.
- El paralelismo lo da Spark (particiones → workers); el modelo HF se carga **una vez por
  executor** (ver `docs/spark-genai-udf.md`), así que escalar = añadir workers, sin tocar código.
- `GENAI_BACKEND=mock` permite validar el flujo de orquestación sin descargar modelos.
