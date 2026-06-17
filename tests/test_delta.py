"""Tests de la persistencia Delta: particionado y time travel."""

import os

from genai_etl.lakehouse.delta_writer import read_delta, table_history, write_delta


def _structured(spark, rows):
    return spark.createDataFrame(rows, schema=["id", "text", "label", "confidence"])


def test_escribe_y_lee_particionado(spark, tmp_path, education_cfg):
    education_cfg.lakehouse.path = str(tmp_path / "lh_edu")
    df = _structured(spark, [("1", "bueno", "POS", 0.9), ("2", "malo", "NEG", 0.7)])

    path = write_delta(df, education_cfg, ingest_date="2026-06-17")
    back = read_delta(spark, path)

    assert back.count() == 2
    assert {"domain", "ingest_date"} <= set(back.columns)
    # directorios de partición físicos
    assert os.path.isdir(os.path.join(path, "domain=education"))
    assert os.path.isdir(os.path.join(path, "domain=education", "ingest_date=2026-06-17"))


def test_time_travel_versionAsOf(spark, tmp_path, education_cfg):
    education_cfg.lakehouse.path = str(tmp_path / "lh_tt")
    education_cfg.lakehouse.mode = "append"

    write_delta(_structured(spark, [("1", "a", "POS", 0.9)]), education_cfg, "2026-06-17")
    write_delta(_structured(spark, [("2", "b", "NEG", 0.8)]), education_cfg, "2026-06-18")

    v0 = read_delta(spark, education_cfg.lakehouse.path, version=0)
    latest = read_delta(spark, education_cfg.lakehouse.path)

    assert v0.count() == 1  # solo la primera corrida
    assert latest.count() == 2  # acumulado tras el append
    assert table_history(spark, education_cfg.lakehouse.path).count() >= 2
