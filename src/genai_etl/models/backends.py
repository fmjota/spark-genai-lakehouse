"""Backends de inferencia intercambiables para las tareas de NLP.

Tres modos seleccionables por la variable de entorno `GENAI_BACKEND`:

- ``mock``  → `MockBackend`: determinista, por reglas de palabras clave, **sin red, sin
  descarga, sin GPU**. Es el backend por defecto en tests/CI (espejo de `HEALER_BACKEND`
  del proyecto 3), de modo que ningún test importe transformers/torch.
- ``hf``    → backends reales que envuelven un `transformers.pipeline`. La importación de
  transformers/torch es **perezosa** (solo al construir el backend), no al importar este
  módulo.
- ``auto``  → usa ``hf`` si transformers está disponible; si no, cae a ``mock``.

Contrato uniforme: `infer(texts) -> list[dict]`, un dict por texto de entrada.

- sentimiento: ``{"label": str, "confidence": float}``
- NER:         ``{"entities": [{"text","type","start","end","score"}], "confidence": float}``

`confidence` está siempre en [0, 1] y alimenta la capa estadística (umbral, drift).
"""

from __future__ import annotations

import hashlib
import os
import re
from abc import ABC, abstractmethod

# --- Léxicos del MockBackend (deterministas, español) --------------------------------

_POS_WORDS = {
    "excelente",
    "bueno",
    "buena",
    "claro",
    "clara",
    "útil",
    "recomiendo",
    "positivo",
    "ganancia",
    "sube",
    "fuerte",
    "crece",
    "favorable",
    "satisfecho",
    "ordenado",
}
_NEG_WORDS = {
    "malo",
    "mala",
    "pésimo",
    "pésima",
    "desorganizado",
    "confuso",
    "caída",
    "pérdida",
    "riesgo",
    "negativo",
    "baja",
    "débil",
    "cae",
    "deficiente",
    "insatisfecho",
    "lento",
}
_DRUGS = {
    "ibuprofeno",
    "paracetamol",
    "amoxicilina",
    "omeprazol",
    "metformina",
    "aspirina",
    "diazepam",
    "losartán",
    "atorvastatina",
    "naproxeno",
}
_SYMPTOMS = {
    "cefalea",
    "dolor",
    "náusea",
    "nausea",
    "fiebre",
    "mareo",
    "erupción",
    "vómito",
    "vomito",
    "fatiga",
    "tos",
    "diarrea",
    "prurito",
}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _stable_unit(text: str) -> float:
    """Pseudo-aleatorio determinista en [0, 1] derivado del texto (reproducible)."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class HFBackend(ABC):
    """Interfaz común de los backends de inferencia."""

    task: str

    @abstractmethod
    def infer(self, texts: list[str], batch_size: int = 16, max_length: int = 256) -> list[dict]:
        """Devuelve una predicción estructurada por cada texto de entrada."""


# --- MockBackend ----------------------------------------------------------------------


class MockSentimentBackend(HFBackend):
    task = "sentiment"

    def infer(self, texts, batch_size=16, max_length=256):
        out = []
        for text in texts:
            tokens = {t.lower() for t in _TOKEN_RE.findall(text or "")}
            pos = len(tokens & _POS_WORDS)
            neg = len(tokens & _NEG_WORDS)
            if pos > neg:
                label = "POS"
            elif neg > pos:
                label = "NEG"
            else:
                label = "NEU"
            # Confianza determinista: base por texto + margen por evidencia de palabras.
            margin = min(abs(pos - neg) * 0.1, 0.3)
            confidence = round(0.6 + 0.3 * _stable_unit(text) + margin, 4)
            out.append({"label": label, "confidence": min(confidence, 0.99)})
        return out


class MockNERBackend(HFBackend):
    task = "ner"

    def infer(self, texts, batch_size=16, max_length=256):
        out = []
        for text in texts:
            entities = []
            for match in _TOKEN_RE.finditer(text or ""):
                word = match.group(0)
                low = word.lower()
                etype = "FARMACO" if low in _DRUGS else "SINTOMA" if low in _SYMPTOMS else None
                if etype is None:
                    continue
                score = round(0.7 + 0.29 * _stable_unit(low), 4)
                entities.append(
                    {
                        "text": word,
                        "type": etype,
                        "start": match.start(),
                        "end": match.end(),
                        "score": score,
                    }
                )
            confidence = (
                round(sum(e["score"] for e in entities) / len(entities), 4) if entities else 0.0
            )
            out.append({"entities": entities, "confidence": confidence})
        return out


# --- Backends reales de Hugging Face (lazy) -------------------------------------------


class HFSentimentBackend(HFBackend):
    task = "sentiment"

    def __init__(self, model_name: str):
        # Importación perezosa: transformers/torch solo se cargan aquí, no al importar el módulo.
        from transformers import pipeline

        self._pipe = pipeline("sentiment-analysis", model=model_name)

    def infer(self, texts, batch_size=16, max_length=256):
        results = self._pipe(
            list(texts),
            batch_size=batch_size,
            truncation=True,
            max_length=max_length,
        )
        return [{"label": r["label"], "confidence": float(r["score"])} for r in results]


class HFNERBackend(HFBackend):
    task = "ner"

    def __init__(self, model_name: str):
        from transformers import pipeline

        self._pipe = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy="simple",
        )

    def infer(self, texts, batch_size=16, max_length=256):
        batch = self._pipe(list(texts), batch_size=batch_size)
        # El pipeline devuelve una lista por texto (o un solo dict si hay un único texto).
        if texts and isinstance(batch, list) and batch and isinstance(batch[0], dict):
            batch = [batch]
        out = []
        for ents in batch:
            entities = [
                {
                    "text": e.get("word", ""),
                    "type": e.get("entity_group", e.get("entity", "ENTIDAD")),
                    "start": int(e.get("start", 0)),
                    "end": int(e.get("end", 0)),
                    "score": float(e.get("score", 0.0)),
                }
                for e in ents
            ]
            confidence = (
                round(sum(e["score"] for e in entities) / len(entities), 4) if entities else 0.0
            )
            out.append({"entities": entities, "confidence": confidence})
        return out


# --- Selección de backend -------------------------------------------------------------


def _transformers_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("transformers") is not None


def build_backend(mode: str, task: str, model_name: str) -> HFBackend:
    """Construye el backend para un modo y tarea dados.

    Args:
        mode: ``mock`` | ``hf`` | ``auto``.
        task: ``sentiment`` | ``ner``.
        model_name: modelo HF para el modo ``hf`` (ignorado en ``mock``).

    Returns:
        HFBackend listo para `infer`.
    """
    if mode == "auto":
        mode = "hf" if _transformers_available() else "mock"

    if mode == "mock":
        return MockSentimentBackend() if task == "sentiment" else MockNERBackend()
    if mode == "hf":
        return HFSentimentBackend(model_name) if task == "sentiment" else HFNERBackend(model_name)
    raise ValueError(f"GENAI_BACKEND desconocido: {mode!r} (usa mock|hf|auto)")


def resolve_mode() -> str:
    """Lee el modo desde `GENAI_BACKEND` (por defecto ``auto``)."""
    return os.environ.get("GENAI_BACKEND", "auto").lower()
