"""Descarga opcional de datasets públicos en español vía Hugging Face `datasets`.

`datasets` no está en las dependencias base (mantiene liviano el entorno de tests/CI). Este
script lo importa de forma perezosa; si falta, indica cómo instalarlo. Normaliza el dataset
al contrato del pipeline (columnas `id`, `texto`) para que el núcleo no distinga datos
reales de sintéticos.

Uso:
    uv run --with datasets python scripts/download_data.py --domain banking --rows 500 \
        --out data/raw/banking_real.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# Dataset público en español sugerido por dominio (sentimiento/reseñas). Editable.
_DATASETS = {
    "education": ("muchocine", "text", None),
    "banking": ("muchocine", "text", None),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga datasets reales en español.")
    parser.add_argument("--domain", required=True, choices=sorted(_DATASETS))
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "Falta `datasets`. Ejecuta con: "
            "uv run --with datasets python scripts/download_data.py ..."
        ) from None

    name, text_col, config = _DATASETS[args.domain]
    ds = load_dataset(name, config, split="train")
    ds = ds.select(range(min(args.rows, len(ds))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "texto"])
        writer.writeheader()
        for i, row in enumerate(ds):
            writer.writerow({"id": str(i), "texto": row[text_col]})
    print(f"Escritas {min(args.rows, len(ds))} filas en {out}")


if __name__ == "__main__":
    main()
