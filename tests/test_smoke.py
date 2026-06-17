"""Smoke test mínimo: el paquete importa y expone su versión."""

import genai_etl


def test_paquete_importa():
    assert genai_etl.__version__
