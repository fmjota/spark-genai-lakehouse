"""Test end-to-end del pipeline multi-dominio en modo mock."""

import sys
from pathlib import Path

import pandas as pd

from genai_etl.enrich.pipeline import run_pipeline
from genai_etl.lakehouse.delta_writer import read_delta

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from generate_synthetic import generate  # noqa: E402


def _write_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)


def test_generate_synthetic_dominios():
    health = generate("health", 20, seed=1)
    edu = generate("education", 20, seed=1)
    assert all("texto" in r for r in health)
    assert all("etiqueta_real" in r for r in edu)  # ground-truth para calibración


def test_pipeline_sentimiento_e2e(spark, tmp_path, education_cfg):
    _write_csv(generate("education", 30, seed=7), tmp_path / "edu.csv")
    education_cfg.data.raw_path = str(tmp_path / "edu.csv")
    education_cfg.lakehouse.path = str(tmp_path / "lh_edu")

    summary = run_pipeline(education_cfg, spark=spark, mode="mock", ingest_date="2026-06-17")

    assert summary["domain"] == "education"
    assert summary["task"] == "sentiment"
    assert summary["n"] == 30
    assert "label_counts" in summary
    # la tabla Delta quedó escrita y es legible
    back = read_delta(spark, summary["delta_path"])
    assert back.count() == 30
    assert {"label", "confidence", "domain", "ingest_date"} <= set(back.columns)


def test_pipeline_ner_e2e(spark, tmp_path, health_cfg):
    _write_csv(generate("health", 30, seed=3), tmp_path / "health.csv")
    health_cfg.data.raw_path = str(tmp_path / "health.csv")
    health_cfg.lakehouse.path = str(tmp_path / "lh_health")

    summary = run_pipeline(health_cfg, spark=spark, mode="mock", ingest_date="2026-06-17")

    assert summary["task"] == "ner"
    assert summary["n"] == 30
    assert "coverage" in summary and 0.0 <= summary["coverage"] <= 1.0
    back = read_delta(spark, summary["delta_path"])
    assert "entities" in back.columns
