#!/usr/bin/env bash
# Equivalente LOCAL de un job de EMR/Dataproc: corre el pipeline con spark-submit.
#
# El mismo código (scripts/run_pipeline.py) corre en `local[*]` y en un cluster cambiando
# solo el master y el aprovisionamiento. Aquí lo lanzamos localmente con los paquetes de
# Delta, apuntando JAVA_HOME a un JDK 17/21 (Spark 4.x no soporta Java 25).
#
# Uso: GENAI_BACKEND=mock iac/local-spark-submit.sh configs/health.yaml
set -euo pipefail

CONFIG="${1:-configs/health.yaml}"
export GENAI_BACKEND="${GENAI_BACKEND:-mock}"

# Descubre un JDK 21 portátil instalado a nivel de usuario.
JDK="$(ls -d "$HOME"/.local/share/jvm/jdk-21* 2>/dev/null | head -1 || true)"
if [ -n "$JDK" ]; then
  export JAVA_HOME="$JDK"
  echo "JAVA_HOME=$JAVA_HOME"
fi

# Versión de Delta acoplada a Spark 4.0 (Scala 2.13).
uv run spark-submit \
  --master "local[*]" \
  --packages io.delta:delta-spark_2.13:4.0.0 \
  --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
  --conf spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog \
  scripts/run_pipeline.py --config "$CONFIG"
