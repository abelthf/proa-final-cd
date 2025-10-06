#!/usr/bin/env python
"""Análisis exploratorio rápido del archivo raw_student_data.csv

Genera:
 - Separador detectado
 - Shape y listado de columnas
 - Tipos de datos inferidos
 - Conteo y porcentaje de valores nulos (top 30)
 - Estadísticos descriptivos numéricos
 - Resumen de cardinalidad de variables categóricas

Uso:
  python quick_eda.py
"""
from __future__ import annotations
import sys
import csv
from pathlib import Path
import pandas as pd

DATA_FILE = Path('raw_student_data.csv')

if not DATA_FILE.exists():
    print(f"[ERROR] No se encuentra el archivo {DATA_FILE.resolve()}", file=sys.stderr)
    sys.exit(1)

def detectar_separador(path: Path) -> str:
    candidatos = [',',';','\t','|']
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        lineas = [next(f) for _ in range(5)]
    for sep in candidatos:
        counts = [ln.count(sep) for ln in lineas]
        if len(set(counts)) == 1 and counts[0] > 0:
            return sep
    # Sniffer fallback
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(4000)
        try:
            dialect = csv.Sniffer().sniff(sample)
            return dialect.delimiter
        except Exception:
            return ','  # por defecto

def cargar_dataframe(path: Path, sep: str) -> pd.DataFrame:
    # Intentar con utf-8, si falla latin-1
    for enc in ('utf-8', 'latin-1'):
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError:
            continue
    # Último intento sin encoding explícito
    return pd.read_csv(path, sep=sep)

def main():
    sep = detectar_separador(DATA_FILE)
    print(f"Separador detectado: {repr(sep)}")
    df = cargar_dataframe(DATA_FILE, sep)
    print(f"Filas x Columnas: {df.shape}")
    print("Columnas:")
    for i, c in enumerate(df.columns, 1):
        print(f"  {i:02d}. {c}")

    print("\nTipos inferidos:")
    print(df.dtypes)

    nulls = df.isna().sum()
    nulls_pct = (nulls/len(df))*100
    nulls_df = (
        pd.DataFrame({'n_nulls': nulls, 'pct_nulls': nulls_pct})
        .sort_values('pct_nulls', ascending=False)
    )
    print("\nValores nulos (top 30 por %):")
    print(nulls_df.head(30))

    # Estadísticos numéricos
    num_cols = df.select_dtypes(include=['number']).columns
    if num_cols.any():
        print("\nEstadísticos numéricos:")
        print(df[num_cols].describe(percentiles=[0.05,0.25,0.5,0.75,0.95]).T)
    else:
        print("\n[INFO] No se detectaron columnas numéricas.")

    # Resumen categóricas
    cat_cols = df.select_dtypes(include=['object','category']).columns
    if cat_cols.any():
        resumen_cat = []
        for c in cat_cols:
            uniques = df[c].nunique(dropna=True)
            top_freq = df[c].value_counts(dropna=True).head(3).to_dict()
            resumen_cat.append({'columna': c, 'n_unique': uniques, 'top_freq': top_freq})
        cat_df = pd.DataFrame(resumen_cat).sort_values('n_unique')
        print("\nResumen variables categóricas (ordenadas por cardinalidad ascendente):")
        print(cat_df.head(30))
    else:
        print("\n[INFO] No se detectaron columnas categóricas.")

    print("\nPreview de datos (primeras 5 filas):")
    with pd.option_context('display.max_columns', 20, 'display.width', 140):
        print(df.head(5))

if __name__ == '__main__':
    main()
