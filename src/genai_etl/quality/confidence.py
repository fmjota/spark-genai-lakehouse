"""Confianza y calibración sobre la salida estructurada.

Cada extracción lleva un `confidence` en [0, 1] (score del modelo). Aquí se resume esa
señal y se discute la **calibración** de forma honesta:

- Sin ground-truth solo se reporta la *distribución* de confianza y la cobertura — no se
  finge que los scores sean probabilidades calibradas.
- Con ground-truth (p.ej. el set sintético trae la etiqueta verdadera) se calcula la
  curva de fiabilidad y el **ECE** (Expected Calibration Error): cuánto se desvía la
  confianza media de la accuracy observada por tramo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def flag_low_confidence(pdf: pd.DataFrame, min_confidence: float) -> pd.DataFrame:
    """Agrega una columna booleana `low_confidence` (confianza < umbral)."""
    out = pdf.copy()
    out["low_confidence"] = out["confidence"] < min_confidence
    return out


def confidence_summary(pdf: pd.DataFrame, min_confidence: float) -> dict:
    """Resume la confianza y la calidad de la salida (agnóstico a la tarea).

    Returns un dict con: n, confianza media, % bajo umbral, y según la tarea la
    distribución de etiquetas (sentimiento) o la cobertura de entidades (NER).
    """
    n = len(pdf)
    conf = pdf["confidence"] if n else pd.Series(dtype=float)
    summary: dict = {
        "n": int(n),
        "mean_confidence": float(conf.mean()) if n else 0.0,
        "pct_low_confidence": float((conf < min_confidence).mean()) if n else 0.0,
    }
    if "label" in pdf.columns:
        summary["label_counts"] = pdf["label"].value_counts().to_dict()
    if "n_entities" in pdf.columns:
        summary["coverage"] = float((pdf["n_entities"] > 0).mean()) if n else 0.0
        summary["total_entities"] = int(pdf["n_entities"].sum())
    return summary


def reliability_table(confidence: pd.Series, correct: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """Tabla de fiabilidad: confianza media vs accuracy observada por tramo de confianza.

    Args:
        confidence: scores en [0, 1].
        correct: booleano/0-1 indicando si la predicción fue correcta (necesita ground-truth).
        n_bins: número de tramos.
    """
    df = pd.DataFrame({"confidence": np.asarray(confidence, dtype=float)})
    df["correct"] = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    df["bin"] = pd.cut(df["confidence"], edges, include_lowest=True)
    grouped = (
        df.groupby("bin", observed=True)
        .agg(
            mean_confidence=("confidence", "mean"),
            accuracy=("correct", "mean"),
            n=("correct", "size"),
        )
        .reset_index()
    )
    return grouped


def expected_calibration_error(
    confidence: pd.Series, correct: pd.Series, n_bins: int = 10
) -> float:
    """ECE: desviación media (ponderada por tramo) entre confianza y accuracy.

    0 = perfectamente calibrado. Requiere ground-truth; sin él, no se debe reportar.
    """
    tbl = reliability_table(confidence, correct, n_bins=n_bins)
    if tbl["n"].sum() == 0:
        return 0.0
    weights = tbl["n"] / tbl["n"].sum()
    return float((weights * (tbl["mean_confidence"] - tbl["accuracy"]).abs()).sum())
