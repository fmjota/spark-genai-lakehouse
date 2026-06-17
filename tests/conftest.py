"""Fixtures compartidas y configuración de tests.

Fija `GENAI_BACKEND=mock` por defecto para que ningún test importe transformers/torch ni
descargue modelos de Hugging Face: el MockBackend es determinista y sin red (espejo del
patrón `HEALER_BACKEND=mock` del proyecto 3).
"""

import os

os.environ.setdefault("GENAI_BACKEND", "mock")

import pytest

from genai_etl.config import load_config


@pytest.fixture(scope="session")
def spark():
    """SparkSession compartida (con Delta) para toda la sesión de tests.

    Se omite si no hay un JDK 17/21 compatible, para no romper en máquinas sin él.
    """
    from genai_etl.spark import find_compatible_java, get_spark

    if find_compatible_java() is None:
        pytest.skip("No hay JDK 17/21 compatible con Spark")
    session = get_spark(app_name="tests", shuffle_partitions=2)
    yield session
    session.stop()


@pytest.fixture
def health_cfg():
    return load_config("configs/health.yaml")


@pytest.fixture
def education_cfg():
    return load_config("configs/education.yaml")


@pytest.fixture
def banking_cfg():
    return load_config("configs/banking.yaml")
