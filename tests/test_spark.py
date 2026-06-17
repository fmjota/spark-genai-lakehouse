"""Tests de la SparkSession (JDK discovery + Delta) y la ingesta de texto."""

from genai_etl.ingest.reader import read_text
from genai_etl.spark import find_compatible_java


def test_descubre_jdk_compatible():
    # En CI hay Java 17; en la laptop del autor hay Temurin 21. Si no, el resto se omite.
    java_home = find_compatible_java()
    if java_home is not None:
        assert "jdk" in java_home.lower() or "java" in java_home.lower()


def test_sesion_tiene_extension_delta(spark):
    ext = spark.conf.get("spark.sql.extensions")
    assert "DeltaSparkSessionExtension" in ext


def test_read_text_normaliza_columnas(spark, tmp_path, education_cfg):
    csv = tmp_path / "education.csv"
    csv.write_text("id,texto\n1,Excelente curso\n2,Muy desorganizado\n", encoding="utf-8")
    education_cfg.data.raw_path = str(csv)

    df = read_text(spark, education_cfg)

    assert df.columns == ["id", "text"]
    rows = {r["id"]: r["text"] for r in df.collect()}
    assert rows == {"1": "Excelente curso", "2": "Muy desorganizado"}


def test_read_text_genera_id_si_falta(spark, tmp_path, education_cfg):
    csv = tmp_path / "no_id.csv"
    csv.write_text("texto\nhola\nchao\n", encoding="utf-8")
    education_cfg.data.raw_path = str(csv)
    education_cfg.data.id_column = None

    df = read_text(spark, education_cfg)

    assert df.columns == ["id", "text"]
    assert df.count() == 2
    assert all(r["id"] is not None for r in df.collect())
