#!/usr/bin/env bash
# Aprovisionamiento de un cluster efímero de GCP Dataproc para correr el pipeline.
#
# DOCUMENTAL: por defecto NO crea nada (DRY_RUN=1). Para ejecutar de verdad:
#   DRY_RUN=0 PROJECT=mi-proyecto BUCKET=mi-bucket iac/dataproc-cluster.sh
#
# Un cluster efímero (se crea, corre el job y se borra) minimiza el costo: se paga solo el
# tiempo del job. El mismo scripts/run_pipeline.py corre aquí sin cambios de código.
set -euo pipefail

PROJECT="${PROJECT:-mi-proyecto}"
REGION="${REGION:-us-central1}"
CLUSTER="${CLUSTER:-genai-etl}"
BUCKET="${BUCKET:-mi-bucket}"
DRY_RUN="${DRY_RUN:-1}"

run() {
  echo "+ $*"
  if [ "$DRY_RUN" = "0" ]; then "$@"; fi
}

# 1) Cluster efímero con inicialización de dependencias Python (transformers, delta-spark).
run gcloud dataproc clusters create "$CLUSTER" \
  --project "$PROJECT" --region "$REGION" \
  --image-version 2.2-debian12 \
  --master-machine-type n2-standard-4 \
  --worker-machine-type n2-standard-4 --num-workers 2 \
  --metadata 'PIP_PACKAGES=transformers torch delta-spark pandera pydantic' \
  --initialization-actions "gs://goog-dataproc-initialization-actions-${REGION}/python/pip-install.sh" \
  --max-idle 30m

# 2) Enviar el job PySpark (Delta vía --properties).
run gcloud dataproc jobs submit pyspark scripts/run_pipeline.py \
  --project "$PROJECT" --region "$REGION" --cluster "$CLUSTER" \
  --properties "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension,spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog" \
  -- --config configs/banking.yaml

# 3) Borrar el cluster (o dejar que --max-idle lo apague).
run gcloud dataproc clusters delete "$CLUSTER" --project "$PROJECT" --region "$REGION" --quiet
