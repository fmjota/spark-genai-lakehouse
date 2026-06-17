"""Generador de datos sintéticos en español, multi-dominio (license-clean).

Produce texto no estructurado plausible para cada dominio, sin PII ni problemas de licencia.
Las plantillas usan el mismo vocabulario que el MockBackend, de modo que el demo en modo
`mock` produzca salida coherente sin descargar modelos. Para sentimiento se incluye la
columna `etiqueta_real` (ground-truth) que habilita la curva de calibración/ECE; el
pipeline la ignora (solo lee `texto`).

Uso:
    uv run python scripts/generate_synthetic.py --domain health --rows 200 --out data/raw/health.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

_FARMACOS = ["ibuprofeno", "paracetamol", "amoxicilina", "omeprazol", "metformina", "naproxeno"]
_SINTOMAS = ["cefalea", "dolor", "náusea", "fiebre", "mareo", "erupción", "fatiga", "tos"]

_HEALTH_CON_ENTIDADES = [
    "El paciente refiere {sintoma} y se le indicó {farmaco}.",
    "Tras administrar {farmaco}, presentó {sintoma} de intensidad leve.",
    "Reporte de evento adverso: {sintoma} posterior a {farmaco}.",
    "Se suspende {farmaco} por {sintoma} persistente.",
]
_HEALTH_SIN_ENTIDADES = [
    "Control de rutina sin hallazgos relevantes.",
    "El paciente acude a su cita de seguimiento habitual.",
    "Se solicita examen de laboratorio de control.",
]

_EDU = {
    "POS": [
        "Excelente curso, el profesor explica muy claro y es muy útil.",
        "Muy buena experiencia, recomiendo el ramo, material ordenado y claro.",
        "Profesor favorable y satisfecho con la calidad de las clases.",
    ],
    "NEG": [
        "Curso pésimo y desorganizado, una pérdida de tiempo.",
        "Material confuso y deficiente, quedé muy insatisfecho.",
        "Clases lentas y mala gestión, alto riesgo de reprobar.",
    ],
    "NEU": [
        "El curso es de matemáticas, se dicta los lunes en la sala 3.",
        "El ramo tiene tres evaluaciones durante el semestre.",
        "Las clases comienzan a las 8 de la mañana.",
    ],
}

_BANK = {
    "POS": [
        "La acción sube fuerte, resultados favorables este trimestre.",
        "Ganancia récord y crecimiento sólido, el mercado reacciona positivo.",
        "Producto excelente, muy recomendable, buena relación calidad precio.",
    ],
    "NEG": [
        "Fuerte caída y pérdida de valor, alto riesgo para el inversionista.",
        "Resultados débiles, la acción baja y el panorama es negativo.",
        "Pésima atención y producto deficiente, no lo recomiendo.",
    ],
    "NEU": [
        "La empresa publicará sus resultados el próximo martes.",
        "El banco informa cambios en el horario de atención.",
        "Se anuncia una reunión de accionistas para el mes entrante.",
    ],
}


def _gen_health(rng: random.Random, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        if rng.random() < 0.75:
            plantilla = rng.choice(_HEALTH_CON_ENTIDADES)
            texto = plantilla.format(sintoma=rng.choice(_SINTOMAS), farmaco=rng.choice(_FARMACOS))
        else:
            texto = rng.choice(_HEALTH_SIN_ENTIDADES)
        rows.append({"id": str(i), "texto": texto})
    return rows


def _gen_sentiment(rng: random.Random, n: int, banco: dict) -> list[dict]:
    rows = []
    etiquetas = list(banco.keys())
    for i in range(n):
        etiqueta = rng.choice(etiquetas)
        rows.append({"id": str(i), "texto": rng.choice(banco[etiqueta]), "etiqueta_real": etiqueta})
    return rows


def generate(domain: str, rows: int, seed: int) -> list[dict]:
    """Genera `rows` filas sintéticas para el dominio dado."""
    rng = random.Random(seed)
    if domain == "health":
        return _gen_health(rng, rows)
    if domain == "education":
        return _gen_sentiment(rng, rows, _EDU)
    if domain == "banking":
        return _gen_sentiment(rng, rows, _BANK)
    raise ValueError(f"Dominio desconocido: {domain} (usa health|education|banking)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generador de datos sintéticos en español.")
    parser.add_argument("--domain", required=True, choices=["health", "education", "banking"])
    parser.add_argument("--rows", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    data = generate(args.domain, args.rows, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)
    print(f"Escritas {len(data)} filas en {out}")


if __name__ == "__main__":
    main()
