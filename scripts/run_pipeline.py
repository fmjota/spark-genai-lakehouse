"""Entrypoint del pipeline: corre un dominio de extremo a extremo y resume en consola.

Uso:
    GENAI_BACKEND=mock uv run python scripts/run_pipeline.py --config configs/health.yaml
    GENAI_BACKEND=hf   uv run python scripts/run_pipeline.py --config configs/banking.yaml
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

from genai_etl.config import load_config
from genai_etl.enrich.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline ETL+GenAI por dominio.")
    parser.add_argument("--config", required=True, help="Ruta al YAML del dominio.")
    parser.add_argument("--mode", default=None, help="Backend: mock|hf|auto (o GENAI_BACKEND).")
    parser.add_argument("--ingest-date", default=None, help="Fecha de ingesta YYYY-MM-DD.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    console = Console()
    console.print(f"[bold]Ejecutando dominio:[/bold] {cfg.domain} ({cfg.model.task})")

    summary = run_pipeline(cfg, mode=args.mode, ingest_date=args.ingest_date)

    table = Table(title=f"Resumen — {cfg.domain}")
    table.add_column("Métrica")
    table.add_column("Valor", justify="right")
    for key in ("n", "mean_confidence", "pct_low_confidence", "coverage", "total_entities"):
        if key in summary:
            val = summary[key]
            table.add_row(key, f"{val:.4f}" if isinstance(val, float) else str(val))
    if "label_counts" in summary:
        table.add_row("label_counts", str(summary["label_counts"]))
    console.print(table)
    console.print(f"[green]Tabla Delta escrita en:[/green] {summary['delta_path']}")


if __name__ == "__main__":
    main()
