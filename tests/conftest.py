"""Fixtures compartidas y configuración de tests.

Fija `GENAI_BACKEND=mock` por defecto para que ningún test importe transformers/torch ni
descargue modelos de Hugging Face: el MockBackend es determinista y sin red (espejo del
patrón `HEALER_BACKEND=mock` del proyecto 3).
"""

import os

os.environ.setdefault("GENAI_BACKEND", "mock")

import pytest

from genai_etl.config import load_config


@pytest.fixture
def health_cfg():
    return load_config("configs/health.yaml")


@pytest.fixture
def education_cfg():
    return load_config("configs/education.yaml")


@pytest.fixture
def banking_cfg():
    return load_config("configs/banking.yaml")
