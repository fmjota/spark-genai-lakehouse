"""Tests de los contratos Pandera de la salida estructurada."""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from genai_etl.schemas import get_output_schema
from genai_etl.validate.runner import validate_output


def test_salida_sentimiento_valida(education_cfg):
    pdf = pd.DataFrame(
        {
            "id": ["1", "2"],
            "text": ["bueno", "malo"],
            "label": ["POS", "NEG"],
            "confidence": [0.9, 0.7],
        }
    )
    out = validate_output(pdf, education_cfg)
    assert len(out) == 2


def test_sentimiento_label_invalido_falla(education_cfg):
    pdf = pd.DataFrame({"id": ["1"], "text": ["x"], "label": ["FELIZ"], "confidence": [0.9]})
    with pytest.raises(SchemaErrors):
        validate_output(pdf, education_cfg)


def test_sentimiento_confianza_fuera_de_rango_falla(education_cfg):
    pdf = pd.DataFrame({"id": ["1"], "text": ["x"], "label": ["POS"], "confidence": [1.5]})
    with pytest.raises(SchemaErrors):
        validate_output(pdf, education_cfg)


def test_salida_ner_valida(health_cfg):
    pdf = pd.DataFrame(
        {
            "id": ["1"],
            "text": ["tomó ibuprofeno"],
            "entities": [
                [{"text": "ibuprofeno", "type": "FARMACO", "start": 5, "end": 15, "score": 0.9}]
            ],
            "n_entities": [1],
            "confidence": [0.9],
        }
    )
    out = validate_output(pdf, health_cfg)
    assert len(out) == 1


def test_ner_tipo_entidad_invalido_falla(health_cfg):
    pdf = pd.DataFrame(
        {
            "id": ["1"],
            "text": ["x"],
            "entities": [[{"text": "x", "type": "LUGAR", "start": 0, "end": 1, "score": 0.9}]],
            "n_entities": [1],
            "confidence": [0.9],
        }
    )
    with pytest.raises(SchemaErrors):
        validate_output(pdf, health_cfg)


def test_get_output_schema_despacha_por_tarea(health_cfg, banking_cfg):
    assert get_output_schema(health_cfg).name == "ner_output"
    assert get_output_schema(banking_cfg).name == "sentiment_output"
