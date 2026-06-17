"""Registry singleton de backends: el modelo se carga **una vez por proceso worker**.

Este es el corazón del patrón anti-cuello-de-botella. Spark ejecuta la UDF en procesos
worker de Python; cada proceso importa este módulo una sola vez. Al cachear el backend en
un diccionario a nivel de módulo, los pesos del modelo se cargan **una vez por executor**,
no por fila ni por partición. Al worker solo viaja el `model_name` (un string), nunca el
objeto del modelo (no se hace broadcast de pesos).
"""

from __future__ import annotations

from genai_etl.models.backends import HFBackend, build_backend, resolve_mode

# Cache a nivel de módulo: persiste mientras viva el proceso worker.
_BACKENDS: dict[str, HFBackend] = {}


def get_backend(task: str, model_name: str, mode: str | None = None) -> HFBackend:
    """Devuelve un backend cacheado por (modo, tarea, modelo), creándolo si no existe.

    Args:
        task: ``sentiment`` | ``ner``.
        model_name: modelo HF (usado solo en modo ``hf``).
        mode: fuerza el modo; si es None se lee de `GENAI_BACKEND`.

    Returns:
        HFBackend reutilizable dentro del proceso actual.
    """
    resolved = (mode or resolve_mode()).lower()
    key = f"{resolved}:{task}:{model_name}"
    if key not in _BACKENDS:
        _BACKENDS[key] = build_backend(resolved, task, model_name)
    return _BACKENDS[key]


def clear_cache() -> None:
    """Vacía el cache (útil en tests para forzar reconstrucción)."""
    _BACKENDS.clear()
