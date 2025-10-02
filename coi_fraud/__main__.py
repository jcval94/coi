
import argparse
import pandas as pd

from . import run_pipeline
from .io.export import export_tables

def main():
    ap = argparse.ArgumentParser(description="COI/Fraud mensual + NLP MX + reportes")
    ap.add_argument("--csv", type=str, required=True, help="Ruta al CSV con columnas base")
    ap.add_argument("--out", type=str, default="./forensic_outputs", help="Directorio de salida")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    reports = run_pipeline(df, language="es")
    paths = export_tables(reports, args.out)
    print("\nExport listo:")
    for key, path in paths.items():
        print(f"  - {key}: {path}")

    # Prueba manual rápida: python -m coi_fraud --csv <archivo.csv> --out ./exports

if __name__ == "__main__":
    main()
