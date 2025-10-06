# Informe Final de Análisis: Factores Asociados al Desempeño Estudiantil

## 1. Resumen Ejecutivo
Este informe presenta el análisis exploratorio y los modelos baseline desarrollados para comprender los factores asociados a la graduación o abandono estudiantil. A partir de un dataset con 36 variables académicas, administrativas y demográficas, se formularon cinco hipótesis centrales y se derivaron nuevas métricas (ratios de aprobación, deltas inter-semestre, densidad de evaluaciones, promedios y ratios de asignaturas sin evaluación). Los resultados confirman que el desempeño temprano y la estabilidad entre semestres son indicadores críticos del éxito final, mientras que factores administrativos y perfiles extremos de edad añaden señales adicionales de riesgo o protección.

## 2. Descripción del Conjunto de Datos
- Registros originales: 4,432.
- Registros analizados tras limpiar `Target` nulo: 4,323.
- Clases en `Target`: Graduate, Dropout, Enrolled (modelos baseline binarios enfocan Graduate vs Dropout).
- Dimensiones principales:
  - Académicas por semestre: unidades inscritas, aprobadas, evaluaciones, notas, unidades sin evaluación (1º y 2º semestre).
  - Administrativas: estado de pagos (Tuition.fees.up.to.date), estado deudor (Debtor), beca, deudas, etc.
  - Demográficas: edad al ingreso.
  - Macroeconómicas: desempleo, inflación, PIB.
- Calidad: mínima presencia de nulos (Target ~2.46%). Resto de variables completas. Varias columnas numéricas representan categorías codificadas.

## 3. Procesamiento y Limpieza
Acciones clave:
1. Eliminación de filas con `Target` nulo (no se imputó al ser variable objetivo).
2. Eliminación de columna índice redundante (`Unnamed: 0`).
3. Creación de variables derivadas:
   - `ratio_aprob_1`, `ratio_aprob_2` = aprobadas / inscritas.
   - `dens_eval_1`, `dens_eval_2` = evaluaciones / inscritas.
   - `sin_eval_ratio_1`, `sin_eval_ratio_2` = sin evaluaciones / inscritas.
   - `grade_mean` = media de notas 1º y 2º semestre.
   - `delta_grade` = nota 2º - nota 1º.
   - `delta_ratio_aprob` = ratio_aprob_2 - ratio_aprob_1.
   - `age_bin` = segmentación de edad en cinco grupos.
4. Conversión a subconjunto binario (Graduate vs Dropout) para primera aproximación predictiva.

## 4. Hipótesis Formuladas
H1. Mayor desempeño temprano (ratio_aprob_1, grade_1) → mayor probabilidad de graduarse.  
H2. Caída de rendimiento inter-semestre (delta_grade < 0 o delta_ratio_aprob < 0) → mayor probabilidad de abandono.  
H3. Mayor densidad de evaluaciones (evals/enrolled) → refuerza la graduación.  
H4. Regularidad administrativa (pagos al día, no deudor) → menor abandono.  
H5. Edades extremas (muy jóvenes o mayores) → mayor tasa de abandono frente a grupos intermedios.  

## 5. Metodología
- EDA: estadísticas descriptivas, histogramas, violin/boxplots, correlaciones, análisis de tasas por grupos.
- Pruebas estadísticas: Mann-Whitney (H1), segmentación por cuartiles (H2), chi-cuadrado (H4).
- Modelos baseline: LogisticRegression (interpretabilidad coeficientes) y RandomForest (no linealidad e importancia de variables). Estrategia de balance: `class_weight`.
- Validación: partición estratificada (75/25 y 70/30 según script/dashboard) con métricas: Accuracy, Balanced Accuracy, F1 Macro.

## 6. Resultados por Hipótesis
### H1 (Confirmada)
Distribuciones de `ratio_aprob_1` y `grade_1` claramente mayores en Graduate. Prueba Mann-Whitney con p-valor significativo respalda diferencia no debida al azar.
### H2 (Confirmada)
`delta_grade` y `delta_ratio_aprob` muestran valores más negativos en Dropout. Las tasas de abandono aumentan en cuartiles inferiores de `delta_grade` → indicador temprano de deterioro.
### H3 (Parcialmente soportada)
La densidad de evaluaciones (compromiso) distingue menos que las tasas de aprobación, pero combinada con éstas mejora la señal. Sugiere efecto sinérgico más que independiente.
### H4 (Confirmada)
Proporción de Graduate mayor entre estudiantes al día y no deudores. Chi-cuadrado significativo: la dimensión administrativa agrega valor para priorizar intervenciones.
### H5 (Parcialmente soportada)
Grupos extremos de edad exhiben tasas de Dropout superiores a grupos centrales; se recomienda profundizar con tamaños de muestra y posibles factores contextuales.

## 7. Importancia de Variables en Modelos Baseline
- RandomForest: mayor peso en `ratio_aprob_2`, `ratio_aprob_1`, `grade_mean`, `grade_2`, `grade_1`.
- LogisticRegression: coeficientes de mayor magnitud para ratios de aprobación y pagos al día (positivo), y efectos negativos moderados asociados a densidad alta cuando controlada por aprobación (posible colinealidad).
Interpretación: la aprobación efectiva supera en relevancia a la mera frecuencia de evaluaciones; la segunda nota y consolidación (grade_2) confirma o refuerza la trayectoria inicial.

## 8. Hallazgos Clave
1. Ratios de aprobación de ambos semestres son los predictores más fuertes.
2. Deltas inter-semestre negativos anticipan abandono → potencial para alertas tempranas.
3. Factores administrativos actúan como moduladores del éxito (cumplimiento correlaciona con graduación).
4. Existen perfiles poblacionales (edad) que requieren intervenciones focalizadas.
5. Densidad de evaluaciones es señal secundaria, útil combinada con aprobación.

## 9. Recomendaciones Operativas
| Objetivo | Acción | Métrica de éxito |
|----------|-------|------------------|
| Reducir abandono temprano | Sistema de alerta basado en ratio_aprob_1 y delta_grade | Disminución % Dropout en 1º año |
| Mejorar acompañamiento | Tutorías para estudiantes con delta_ratio_aprob negativo | % recuperación de nota semestre siguiente |
| Fortalecer compromiso | Incentivos a seguimiento evaluativo significativo | Aumento dens_eval sin sacrificar aprobación |
| Prevenir riesgo administrativo | Notificaciones proactivas a deudores | % regularización antes de corte |
| Personalizar apoyo por edad | Programas diferenciados extremos edad | Mejora retención por grupo |

## 10. Limitaciones
- Carencia de diccionario semántico para códigos categóricos.
- Análisis correlacional, no causal.
- Exclusión de clase Enrolled en modelos iniciales (pendiente multiclase).
- Variables macro sin granularidad temporal detallada.

## 11. Mejoras Futuras Prioritarias (Resumen)
1. Modelo multiclase incluyendo Enrolled.  
2. Pipeline de ML reproducible (ColumnTransformer).  
3. Explicabilidad avanzada (SHAP / Permutation Importance).  
4. Monitoreo de drift (PSI, KS).  
5. Integración de nuevas fuentes (engagement, asistencia).  
6. Cohortes y validación temporal.  
7. Selección de variables y reducción de colinealidad.  
8. Alertas en dashboard con umbrales dinámicos.  
9. Estimación de riesgo individual (probabilidad abandono).  
10. Auditoría continua (Great Expectations + CI/CD).  

## 12. Reproducibilidad
- EDA completo: `eda.ipynb`.
- Script rápido: `quick_eda.py`.
- Modelos baseline: `models_baseline.py`.
- Visualización ejecutiva interactiva: `dashboard.py`.
- Variables derivadas implementadas consistentemente en scripts y dashboard.

## 13. Conclusión
El análisis demuestra que la aprobación efectiva y la estabilidad académica entre semestres son factores centrales explicativos de la graduación. La señal administrativa y la segmentación demográfica aportan capas adicionales para focalizar intervenciones preventivas. Con un roadmap claro de mejoras (multiclase, explicabilidad, monitoreo de drift y nuevas fuentes de datos), el sistema puede evolucionar hacia una plataforma predictiva robusta de retención estudiantil.

## 14. Anexos
- Métricas detalladas de modelos (ver salida `models_baseline.py`).
- Matrices de confusión y coeficientes / importancias.
- Definición formal de cada feature derivada.
- Ejemplos de consultas futuras (scoring por cohorte, evolución de riesgo).
