#!/usr/bin/env python
"""Modelos baseline para predecir Target.

Entrena LogisticRegression y RandomForest utilizando variables derivadas clave.
Genera:
 - Métricas (Accuracy, F1 macro, Balanced Accuracy)
 - Matriz de confusión
 - Importancia de características (coeficientes y feature_importances_)
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report

DATA_PATH = Path('raw_student_data.csv')
assert DATA_PATH.exists(), 'No se encuentra raw_student_data.csv'

def cargar_base() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df[~df['Target'].isna()].copy()
    rename_map = {
        'Curricular.units.1st.sem..enrolled.': 'enrolled_1',
        'Curricular.units.1st.sem..approved.': 'approved_1',
        'Curricular.units.1st.sem..evaluations.': 'evals_1',
        'Curricular.units.1st.sem..grade.': 'grade_1',
        'Curricular.units.1st.sem..without.evaluations.': 'without_eval_1',
        'Curricular.units.2nd.sem..enrolled.': 'enrolled_2',
        'Curricular.units.2nd.sem..approved.': 'approved_2',
        'Curricular.units.2nd.sem..evaluations.': 'evals_2',
        'Curricular.units.2nd.sem..grade.': 'grade_2',
        'Curricular.units.2nd.sem..without.evaluations.': 'without_eval_2'
    }
    df = df.rename(columns=rename_map)
    for num, den, new_name in [
        ('approved_1','enrolled_1','ratio_aprob_1'),
        ('approved_2','enrolled_2','ratio_aprob_2'),
        ('evals_1','enrolled_1','dens_eval_1'),
        ('evals_2','enrolled_2','dens_eval_2'),
        ('without_eval_1','enrolled_1','sin_eval_ratio_1'),
        ('without_eval_2','enrolled_2','sin_eval_ratio_2')
    ]:
        df[new_name] = np.where(df[den] > 0, df[num]/df[den], np.nan)
    df['grade_mean'] = df[['grade_1','grade_2']].mean(axis=1)
    df['delta_grade'] = df['grade_2'] - df['grade_1']
    df['delta_ratio_aprob'] = df['ratio_aprob_2'] - df['ratio_aprob_1']
    return df

def preparar_dataset(df: pd.DataFrame):
    # Filtrar solo Graduate vs Dropout para clasificación binaria inicial
    subset = df[df['Target'].isin(['Graduate','Dropout'])].copy()
    subset['Target_bin'] = (subset['Target']=='Graduate').astype(int)
    features = [
        'ratio_aprob_1','ratio_aprob_2','grade_1','grade_2','grade_mean','delta_grade',
        'delta_ratio_aprob','dens_eval_1','dens_eval_2','sin_eval_ratio_1','sin_eval_ratio_2',
        'Tuition.fees.up.to.date','Debtor','Age.at.enrollment'
    ]
    X = subset[features].fillna(0)
    y = subset['Target_bin']
    return X, y, features

def entrenar_modelos(X, y, features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

    # Logistic Regression pipeline manual (escala numérica)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    logit = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=None)
    logit.fit(X_train_s, y_train)
    y_pred_log = logit.predict(X_test_s)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=400, random_state=42, class_weight='balanced_subsample', n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    def metricas(nombre, y_true, y_pred):
        return {
            'modelo': nombre,
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro')
        }

    resultados = [
        metricas('LogisticRegression', y_test, y_pred_log),
        metricas('RandomForest', y_test, y_pred_rf)
    ]

    print('Métricas:')
    print(pd.DataFrame(resultados))
    print('\nMatriz de confusión (LogisticRegression):')
    print(confusion_matrix(y_test, y_pred_log))
    print('\nMatriz de confusión (RandomForest):')
    print(confusion_matrix(y_test, y_pred_rf))
    print('\nReporte clasificación RF:')
    print(classification_report(y_test, y_pred_rf, target_names=['Dropout','Graduate']))

    # Importancias / coeficientes
    coef = pd.Series(logit.coef_[0], index=features).sort_values(key=abs, ascending=False)
    imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print('\nCoeficientes (ordenados por magnitud) - LogisticRegression:')
    print(coef.head(15))
    print('\nImportancia de características - RandomForest:')
    print(imp.head(15))

def main():
    df = cargar_base()
    X, y, features = preparar_dataset(df)
    entrenar_modelos(X, y, features)

if __name__ == '__main__':
    main()
