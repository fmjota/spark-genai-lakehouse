"""Tests de los backends de inferencia y el registry singleton (modo mock)."""

from genai_etl.models import registry
from genai_etl.models.backends import (
    MockNERBackend,
    MockSentimentBackend,
    build_backend,
)


def test_mock_sentiment_es_determinista():
    b = MockSentimentBackend()
    r1 = b.infer(["Excelente curso, muy claro y útil"])
    r2 = b.infer(["Excelente curso, muy claro y útil"])
    assert r1 == r2  # determinismo
    assert r1[0]["label"] == "POS"
    assert 0.0 <= r1[0]["confidence"] <= 1.0


def test_mock_sentiment_detecta_negativo():
    b = MockSentimentBackend()
    out = b.infer(["Curso pésimo y desorganizado, una pérdida de tiempo"])
    assert out[0]["label"] == "NEG"


def test_mock_sentiment_neutro_sin_evidencia():
    b = MockSentimentBackend()
    out = b.infer(["El curso es de matemáticas en la sala 3"])
    assert out[0]["label"] == "NEU"


def test_mock_ner_extrae_farmacos_y_sintomas():
    b = MockNERBackend()
    out = b.infer(["El paciente tomó ibuprofeno por la cefalea y tuvo náusea"])
    tipos = {e["type"] for e in out[0]["entities"]}
    textos = {e["text"].lower() for e in out[0]["entities"]}
    assert "FARMACO" in tipos and "SINTOMA" in tipos
    assert "ibuprofeno" in textos
    # confianza = promedio de scores de entidades, en [0, 1]
    assert 0.0 < out[0]["confidence"] <= 1.0


def test_mock_ner_sin_entidades_confianza_cero():
    b = MockNERBackend()
    out = b.infer(["El cielo está despejado hoy"])
    assert out[0]["entities"] == []
    assert out[0]["confidence"] == 0.0


def test_spans_apuntan_al_texto():
    b = MockNERBackend()
    texto = "tomó ibuprofeno ayer"
    out = b.infer([texto])
    e = out[0]["entities"][0]
    assert texto[e["start"] : e["end"]] == e["text"]


def test_build_backend_mock_por_tarea():
    assert isinstance(build_backend("mock", "sentiment", "x"), MockSentimentBackend)
    assert isinstance(build_backend("mock", "ner", "x"), MockNERBackend)


def test_auto_cae_a_mock_sin_transformers(monkeypatch):
    import genai_etl.models.backends as backends

    monkeypatch.setattr(backends, "_transformers_available", lambda: False)
    b = backends.build_backend("auto", "sentiment", "x")
    assert isinstance(b, MockSentimentBackend)


def test_registry_cachea_misma_instancia():
    registry.clear_cache()
    b1 = registry.get_backend("ner", "modelo-x", mode="mock")
    b2 = registry.get_backend("ner", "modelo-x", mode="mock")
    assert b1 is b2  # singleton: no se reconstruye


def test_registry_distingue_por_tarea():
    registry.clear_cache()
    b_ner = registry.get_backend("ner", "m", mode="mock")
    b_sent = registry.get_backend("sentiment", "m", mode="mock")
    assert b_ner is not b_sent
