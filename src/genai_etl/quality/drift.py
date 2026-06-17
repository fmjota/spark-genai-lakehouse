"""Drift de la distribución de texto y de etiquetas entre un lote de referencia y uno nuevo.

Reutiliza la idea de monitoreo de P1 (PSI/KS) aplicada al texto: si la distribución de
longitudes del texto o la mezcla de etiquetas predichas se desplaza, conviene desconfiar de
la salida antes de que degrade silenciosamente. PSI mide desplazamiento de distribución; KS
contrasta si dos muestras provienen de la misma distribución continua.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

_EPS = 1e-6


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index entre dos muestras numéricas (mismos cortes por cuantil)."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e = np.where(e == 0, _EPS, e)
    a = np.where(a == 0, _EPS, a)
    return float(np.sum((a - e) * np.log(a / e)))


def categorical_psi(ref: pd.Series, new: pd.Series) -> float:
    """PSI categórico sobre proporciones de etiquetas (unión de categorías de ambos lotes)."""
    ref_p = ref.value_counts(normalize=True)
    new_p = new.value_counts(normalize=True)
    cats = set(ref_p.index) | set(new_p.index)
    total = 0.0
    for c in cats:
        e = max(float(ref_p.get(c, 0.0)), _EPS)
        a = max(float(new_p.get(c, 0.0)), _EPS)
        total += (a - e) * np.log(a / e)
    return float(total)


def text_length_drift(ref_texts: pd.Series, new_texts: pd.Series) -> dict:
    """Drift de la longitud del texto (en caracteres): PSI + test KS de dos muestras."""
    ref_len = ref_texts.fillna("").str.len().to_numpy()
    new_len = new_texts.fillna("").str.len().to_numpy()
    ks = ks_2samp(ref_len, new_len)
    return {
        "psi": psi(ref_len, new_len),
        "ks_stat": float(ks.statistic),
        "ks_pvalue": float(ks.pvalue),
    }


def drift_report(ref_pdf: pd.DataFrame, new_pdf: pd.DataFrame, cfg) -> dict:
    """Reporte de drift entre referencia y nuevo lote, con banderas según los umbrales.

    Incluye drift de longitud de texto siempre, y de distribución de etiquetas si la tarea
    es de sentimiento. `drift_detectado` resume si algún indicador cruza su umbral.
    """
    report = {"text_length": text_length_drift(ref_pdf["text"], new_pdf["text"])}

    if "label" in ref_pdf.columns and "label" in new_pdf.columns:
        report["label_psi"] = categorical_psi(ref_pdf["label"], new_pdf["label"])

    length_drift = (
        report["text_length"]["psi"] > cfg.quality.psi_threshold
        or report["text_length"]["ks_pvalue"] < cfg.quality.ks_pvalue_threshold
    )
    label_drift = report.get("label_psi", 0.0) > cfg.quality.psi_threshold
    report["drift_detectado"] = bool(length_drift or label_drift)
    return report
