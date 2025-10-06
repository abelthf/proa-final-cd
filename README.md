# Análisis de Desempeño Estudiantil

> Trabajo del curso **Ciencia de Datos y Big Data** de la **Maestría en Ciencia de la Computación** de la **Universidad Nacional del Altiplano**.

## Descripción
Proyecto de analítica académica para identificar factores asociados a graduación, abandono e inscripción continua. La versión actual consolida el flujo en el cuaderno principal `main.ipynb`, que integra:
- Limpieza y estandarización de variables.
- Construcción de métricas derivadas (ratios, deltas, scores administrativos).
- Evaluación robusta de 3 hipótesis (H1–H3) con pruebas no paramétricas y tamaños de efecto.
- Generación de figuras definitivas y conclusiones accionables.

Recursos complementarios:
- `eda.ipynb`: exploración inicial y trazabilidad histórica (etapas previas, ahora resumido en `main.ipynb`).
- `quick_eda.py`: inspección rápida de estructura/nulos.
- `models_baseline.py`: modelos baseline (LogisticRegression / RandomForest) para referencia.
- `dashboard.py`: panel ejecutivo (Streamlit) focalizado en H1–H3 y calidad de datos.
- `final_report.md` / `final_report_outline.md`: narrativa y estructura del informe final.

## Estructura Principal (carpetas y archivos clave)
```
raw_student_data.csv   # Datos originales
main.ipynb             # Cuaderno principal (flujo consolidado H1–H3)
eda.ipynb              # Versión exploratoria anterior (referencia histórica)
quick_eda.py           # Resumen rápido de estructura y nulos
models_baseline.py     # Modelos baseline
dashboard.py           # Dashboard Streamlit
final_report_outline.md# Outline del informe final
env/                   # Entorno virtual (opcional)
```

## Requerimientos
Python 3.11+ (probado en 3.13).

Paquetes principales:
```
pandas numpy plotly streamlit scipy scikit-learn
```
Adicionales recomendados (para extensiones o análisis complementario):
```
seaborn statsmodels ipykernel
```
Instalación rápida:
```
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## Uso del Cuaderno Principal (`main.ipynb`)
1. Abrir `main.ipynb` y ejecutar secuencialmente (las secciones están ordenadas: Setup -> Limpieza -> Métricas -> H1 -> H2 -> H3 -> Conclusiones).
2. Verificar tablas de resumen y figuras definitivas (cada figura tiene caption descriptivo en H2 y H3).
3. Ajustar rutas si se mueve el CSV fuente (`raw_student_data.csv`).
4. (Opcional) Regenerar dataset enriquecido exportando desde las celdas de síntesis.

Para una inspección rápida en terminal:
```
python quick_eda.py
```

## Modelos Baseline
```
python models_baseline.py
```
Salida: métricas (accuracy, balanced_accuracy, f1_macro), matriz de confusión e importancia de variables.

## Dashboard Interactivo (`dashboard.py`)
Ejecutar:
```
streamlit run dashboard.py
```
Características actuales:
- Pestañas H1 (desempeño temprano), H2 (deterioro inter-semestre), H3 (factores administrativos), Calidad de Datos y Conclusiones.
- Filtros de estado (Graduado / Abandono / Inscrito) y variables administrativas clave.
- Cálculos de Kruskal-Wallis, Dunn post-hoc (BH-FDR) y Chi² + Cramér's V (corregido) embebidos.
- Métricas derivadas (ratios y deltas) y score administrativo heurístico.
- Descarga de dataset enriquecido (alias + derivadas).

## Hipótesis Actuales (Consolidadas)
1. H1: El desempeño académico temprano (Semestre 1) diferencia el estado final (graduación / abandono / inscripción).
2. H2: El deterioro inter-semestre (cambios negativos en nota y ratio de aprobación) está asociado al abandono.
3. H3: Factores administrativos (pagos, mora, becas, regularidad) modulan la probabilidad de graduación.

Hipótesis anteriores (H4/H5 originales) se integraron conceptualmente en H3 (administrativo) y en filtros exploratorios; se documentan en versiones previas (`eda.ipynb`).

## Variables Derivadas Clave
- ratio_aprob_1 / ratio_aprob_2
- grade_1 / grade_2 / grade_mean
- delta_grade, delta_ratio_aprob
- dens_eval_1 / dens_eval_2 (si columnas de evaluaciones originales disponibles)
- admin_score_sum (suma heurística de indicadores administrativos favorables)

## Reproducibilidad
La trazabilidad histórica está en `eda.ipynb`, mientras que la versión consolidada y final se encuentra en `main.ipynb`. El feature engineering aplicado al dataset operativo se replica en el dashboard y scripts auxiliares.

## Próximos Pasos (Sugeridos)
- Incorporar interpretabilidad (SHAP / Permutation Importance) sobre el modelo final multiclase.
- Integrar monitor de drift si se añaden nuevas cohortes.
- Formalizar pipeline de imputación y documentación de codificaciones categóricas.
- Evaluar modelos adicionales (XGBoost / LightGBM) y calibración de probabilidades.
- Añadir pruebas unitarias ligeras para validación de derivadas clave.

## Licencia
Uso académico / educativo.

