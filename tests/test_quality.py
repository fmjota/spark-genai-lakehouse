"""Tests de la capa estadística: confianza/calibración y drift."""

import numpy as np
import pandas as pd

from genai_etl.quality.confidence import (
    confidence_summary,
    expected_calibration_error,
    flag_low_confidence,
)
from genai_etl.quality.drift import categorical_psi, drift_report, psi, text_length_drift

# --- Confianza ---


def test_flag_low_confidence():
    pdf = pd.DataFrame({"confidence": [0.9, 0.3, 0.6]})
    out = flag_low_confidence(pdf, min_confidence=0.5)
    assert out["low_confidence"].tolist() == [False, True, False]


def test_confidence_summary_sentimiento():
    pdf = pd.DataFrame({"label": ["POS", "POS", "NEG"], "confidence": [0.9, 0.4, 0.8]})
    s = confidence_summary(pdf, min_confidence=0.5)
    assert s["n"] == 3
    assert s["label_counts"]["POS"] == 2
    assert 0.0 <= s["pct_low_confidence"] <= 1.0


def test_confidence_summary_ner_cobertura():
    pdf = pd.DataFrame({"n_entities": [0, 2, 1], "confidence": [0.0, 0.9, 0.8]})
    s = confidence_summary(pdf, min_confidence=0.5)
    assert s["coverage"] == 2 / 3
    assert s["total_entities"] == 3


def test_ece_perfectamente_calibrado_es_cero():
    # confianza = accuracy en cada caso -> ECE ~ 0
    conf = pd.Series([0.95] * 100)
    correct = pd.Series([1] * 95 + [0] * 5)
    ece = expected_calibration_error(conf, correct, n_bins=10)
    assert ece < 0.05


def test_ece_detecta_sobreconfianza():
    conf = pd.Series([0.99] * 100)
    correct = pd.Series([1] * 50 + [0] * 50)  # solo 50% acierto
    ece = expected_calibration_error(conf, correct, n_bins=10)
    assert ece > 0.4


# --- Drift ---


def test_psi_cero_misma_distribucion():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1000)
    assert psi(x, x) < 0.01


def test_psi_alto_distribuciones_distintas():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 1000)
    b = rng.normal(5, 1, 1000)
    assert psi(a, b) > 0.25


def test_categorical_psi_detecta_cambio_de_mezcla():
    ref = pd.Series(["POS"] * 80 + ["NEG"] * 20)
    new = pd.Series(["POS"] * 20 + ["NEG"] * 80)
    assert categorical_psi(ref, new) > 0.25


def test_text_length_drift():
    ref = pd.Series(["hola"] * 50)
    new = pd.Series(["un texto bastante más largo que el de referencia"] * 50)
    d = text_length_drift(ref, new)
    assert d["psi"] >= 0.0
    assert d["ks_pvalue"] < 0.05  # distribuciones de longitud claramente distintas


def test_drift_report_flag(education_cfg):
    ref = pd.DataFrame({"text": ["bueno"] * 50, "label": ["POS"] * 50})
    new = pd.DataFrame({"text": ["malo"] * 50, "label": ["NEG"] * 50})
    rep = drift_report(ref, new, education_cfg)
    assert "text_length" in rep and "label_psi" in rep
    assert rep["drift_detectado"] is True
