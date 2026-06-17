"""Tests del cargador de config de dominio."""

import pytest
from pydantic import ValidationError

from genai_etl.config import DomainConfig, load_config


@pytest.mark.parametrize(
    "path",
    ["configs/health.yaml", "configs/education.yaml", "configs/banking.yaml"],
)
def test_carga_configs_validos(path):
    cfg = load_config(path)
    assert isinstance(cfg, DomainConfig)
    assert cfg.domain
    assert cfg.data.text_column
    assert cfg.model.task in ("sentiment", "ner")


def test_health_es_ner(health_cfg):
    assert health_cfg.model.task == "ner"
    assert health_cfg.model.entity_types  # define tipos de entidad esperados


def test_sentiment_define_labels(education_cfg, banking_cfg):
    for cfg in (education_cfg, banking_cfg):
        assert cfg.model.task == "sentiment"
        assert cfg.model.labels


def test_particionado_por_dominio_y_fecha(health_cfg):
    assert health_cfg.lakehouse.partition_by == ["domain", "ingest_date"]


def test_config_invalido_falla():
    # task fuera del Literal permitido y campos requeridos ausentes
    raw = {
        "domain": "x",
        "data": {"raw_path": "d.csv", "text_column": "t"},
        "model": {"task": "translation", "model_name": "m"},
        "lakehouse": {"path": "lh"},
    }
    with pytest.raises(ValidationError):
        DomainConfig.model_validate(raw)


def test_min_confidence_fuera_de_rango_falla():
    raw = {
        "domain": "x",
        "data": {"raw_path": "d.csv", "text_column": "t"},
        "model": {"task": "ner", "model_name": "m"},
        "lakehouse": {"path": "lh"},
        "quality": {"min_confidence": 1.5},
    }
    with pytest.raises(ValidationError):
        DomainConfig.model_validate(raw)
