# Informe Final - Análisis de Desempeño Estudiantil

## 1. Resumen Ejecutivo
Este informe analiza factores asociados al resultado académico final (Target: Graduate, Dropout, Enrolled) usando 36 variables académicas, socio-administrativas y macroeconómicas. Se formularon 5 hipótesis y se desarrollaron métricas derivadas para evaluar su validez. Se identifican indicadores tempranos de riesgo de abandono y palancas de mejora.

## 2. Conjunto de Datos
- Registros: 4,432 (tras limpieza: sin nulos en Target, 4,323).
- Columnas: métricas por semestre (inscritas, aprobadas, evaluaciones, notas), variables administrativas (Tuition.fees.up.to.date, Debtor), demográficas (Age.at.enrollment), contexto macroeconómico.
- Calidad: Muy pocos nulos (solo en Target ~2.46%). Codificación numérica de categorías sin diccionario explícito.

## 3. Limpieza y Preparación
- Eliminación de filas con Target nulo.
- Eliminación de columna índice redundante.
- Creación de variables derivadas: ratios de aprobación, densidad de evaluaciones, variaciones inter-semestre, promedio de notas, ratio de asignaturas sin evaluación, bins de edad.

## 4. Hipótesis Formuladas
H1: Mayor desempeño temprano (ratio_aprob_1, grade_1) -> mayor probabilidad de Graduate.
H2: Caída en rendimiento (delta_grade < 0 o delta_ratio_aprob < 0) -> incrementa Dropout.
H3: Mayor densidad de evaluaciones (evals/enrolled) -> mayor probabilidad de Graduate.
H4: Estar al día en pagos y no ser deudor -> mayor graduación y menor abandono.
H5: Edades extremas al ingreso (muy jóvenes o mayores) -> mayor tasa de Dropout.

## 5. Metodología Analítica
- Estadísticos descriptivos y visualizaciones (distribuciones, violin/boxplots, heatmap correlaciones, tasas por grupos).
- Pruebas estadísticas: Mann-Whitney (H1), segmentación por cuartiles (H2), chi-cuadrado (H4).
- Modelos baseline (LogisticRegression, RandomForest) para estimar importancia de variables académicas y administrativas.

## 6. Resultados por Hipótesis (Síntesis)
H1: Diferencias claras en ratio_aprob_1 y grade_1 entre Graduate y Dropout (p-valor Mann-Whitney significativo). CONFIRMADA.
H2: Tasas de Dropout mayores en cuartiles con delta_grade negativa pronunciada. CONFIRMADA.
H3: Densidad de evaluaciones muestra correlación moderada con graduación, pero menor que ratios de aprobación. PARCIALMENTE SOPORTADA.
H4: Estar al día en pagos asociado positivamente a Graduate; chi-cuadrado significativo. CONFIRMADA.
H5: Grupos extremos de edad presentan mayor tasa de Dropout que grupos centrales. PARCIALMENTE SOPORTADA (revisar tamaño de muestra en extremos).

## 7. Importancia de Variables (Modelos)
- RandomForest destaca: ratio_aprob_2, ratio_aprob_1, grade_mean, grade_2, grade_1.
- Variables administrativas (Tuition.fees.up.to.date, Debtor) relevantes pero secundarias frente a desempeño.
- LogisticRegression: coeficientes positivos fuertes para ratios de aprobación; signo esperado para pagos.

## 8. Hallazgos Clave
1. Indicadores tempranos de aprobación son el mejor predictor del resultado final.
2. Caídas entre semestres son señales de alerta oportunas.
3. Compromiso (densidad de evaluaciones) influye pero está mediado por calidad (aprobaciones).
4. La gestión administrativa (pagos al día) es un factor de soporte del éxito académico.
5. Perfiles de edad extremos requieren intervenciones personalizadas.

## 9. Recomendaciones
- Implementar sistema de alerta temprana usando ratios de aprobación y delta_grade.
- Programa de tutorías focalizado en estudiantes con caída inter-semestre.
- Incentivos para mantener regularidad administrativa (evitar morosidad).
- Segmentar estrategias de retención por grupo de edad.
- Enriquecer dataset con: historial previo, engagement en plataformas, asistencia.

## 10. Limitaciones
- Falta diccionario de códigos (reduce interpretabilidad semántica de algunas variables).
- No se midió causalidad, solo asociación.
- Clases 'Enrolled' no incluidas en modelo binario inicial (podría incorporarse modelo multiclase).

## 11. Próximos Pasos Técnicos
### 11.1 Mejoras Futuras Prioritarias
1. Clasificación Multiclase Completa: Incluir clase 'Enrolled' con técnicas de manejo de clases desbalanceadas (e.g., focal loss, class weights).
2. Enriquecimiento Semántico: Incorporar diccionario de códigos para variables categóricas codificadas numéricamente y facilitar interpretabilidad ejecutiva.
3. Pipeline Productizable: Construir un pipeline scikit-learn (ColumnTransformer + imputación + escalado + codificación) y empaquetar con `joblib`.
4. Monitoreo de Drift: Implementar métricas de estabilidad (PSI, KS) para detectar cambios en distribución de variables clave.
5. Explicabilidad Avanzada: Aplicar SHAP / Permutation Importance sobre modelos de árbol y comparar con coeficientes logísticos.
6. Curvas de Aprendizaje y Validación Temporal: Evaluar sobre múltiples cortes cronológicos para robustez fuera de muestra.
7. Feature Engineering Adicional: Ratios acumulados, flags de estancamiento (evaluaciones sin mejora), índice de esfuerzo (evaluations/enrolled ajustado por aprobaciones).
8. Detección de Outliers: IsolationForest o IQR robusto para excluir valores atípicos que distorsionan notas y ratios.
9. Análisis de Cohortes: Segmentar por año de ingreso y comparar evolución de métricas de riesgo.
10. Integración de Datos Externos: Tasas macro desagregadas por periodo real de inscripción; asistencia, interacción LMS, tutorías.

### 11.2 Roadmap Sugerido (0-12 semanas)
- Semanas 1-2: Diccionario de códigos + pipeline reproducible.
- Semanas 3-4: Modelo multiclase + métricas ampliadas (ROC micro/macro, PR curves).
- Semanas 5-6: Explicabilidad (SHAP) + dashboard de riesgo con alertas tempranas.
- Semanas 7-8: Monitoreo de drift y logging de predicciones.
- Semanas 9-10: Integración de nuevas fuentes (asistencia / LMS).
- Semanas 11-12: End-to-end CI/CD (tests, validación de datos con Great Expectations).

## 12. Reproducibilidad
Ver notebook `eda.ipynb` y scripts `quick_eda.py`, `models_baseline.py`.

## 13. Anexos
- Tablas de métricas completas.
- Matrices de confusión.
- Definiciones formales de variables derivadas.
