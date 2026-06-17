"""Fábrica de SparkSession con descubrimiento de un JDK compatible y extensión Delta Lake.

Spark 4.x soporta Java 17/21, no Java 25. En esta máquina (Fedora) el Java por defecto es
25, así que apuntamos `JAVA_HOME` a un JDK 17/21 dedicado **solo para el proceso de Spark**,
sin tocar el Java del sistema. Buscamos primero en el entorno y luego en un JDK instalado a
nivel de usuario en `~/.local/share/jvm`.

La sesión se crea con la extensión de Delta Lake habilitada (`configure_spark_with_delta_pip`
añade los jars de Delta que correspondan a la versión instalada de `delta-spark`), de modo
que `spark.read.format("delta")` y `df.write.format("delta")` funcionan sin más setup.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

_COMPATIBLE_MAJORS = (21, 17)


def _java_major(java_home: str) -> int | None:
    """Lee la versión mayor de Java desde el archivo `release` del JDK."""
    release = Path(java_home) / "release"
    if not release.exists():
        return None
    for line in release.read_text().splitlines():
        if line.startswith("JAVA_VERSION="):
            ver = line.split("=", 1)[1].strip().strip('"')
            parts = ver.split(".")
            # "21.0.11" -> 21 ; "1.8.0_xx" -> 8
            return int(parts[1]) if parts[0] == "1" else int(parts[0])
    return None


def find_compatible_java() -> str | None:
    """Devuelve la ruta de un JDK 17/21 compatible con Spark, o None."""
    env = os.environ.get("JAVA_HOME")
    if env and _java_major(env) in _COMPATIBLE_MAJORS:
        return env
    candidates: list[str] = []
    for major in _COMPATIBLE_MAJORS:
        candidates += glob.glob(str(Path.home() / f".local/share/jvm/jdk-{major}*"))
    for path in sorted(candidates, reverse=True):
        if _java_major(path) in _COMPATIBLE_MAJORS:
            return path
    return None


def ensure_java_home() -> str:
    """Fija JAVA_HOME a un JDK compatible; falla temprano y claro si no hay ninguno."""
    java_home = find_compatible_java()
    if java_home is None:
        raise RuntimeError(
            "No se encontró un JDK 17/21 compatible con Spark. Instala uno "
            "(p.ej. Temurin 21 en ~/.local/share/jvm) o exporta JAVA_HOME a un JDK 17/21."
        )
    os.environ["JAVA_HOME"] = java_home
    return java_home


def get_spark(app_name: str = "genai-etl", shuffle_partitions: int = 8) -> SparkSession:
    """Crea (o reutiliza) una SparkSession local con JAVA_HOME garantizado y Delta habilitado.

    Args:
        app_name: nombre de la aplicación Spark.
        shuffle_partitions: particiones de shuffle (bajar en local acelera los tests).

    Returns:
        SparkSession con la extensión Delta Lake activa.

    Side effects:
        Fija `os.environ["JAVA_HOME"]` a un JDK compatible. La primera creación puede
        descargar los jars de Delta desde Maven.
    """
    ensure_java_home()
    builder = (
        SparkSession.builder.appName(app_name)
        .master(os.environ.get("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
    )
    return configure_spark_with_delta_pip(builder).getOrCreate()
