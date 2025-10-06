#!/usr/bin/env python
"""Dashboard ejecutivo (Streamlit) - Evaluación de 3 hipótesis clave.

Ejecutar:
  streamlit run dashboard.py

Objetivo: Proveer al Director Ejecutivo una visión inmediata sobre:
 - H1: Desempeño académico temprano y su poder predictivo.
 - H2: Deterioro inter-semestre y su asociación con abandono.
 - H3: Factores administrativos y su modulación del resultado.
Además se resumen sorpresas y desafíos de calidad de datos.
"""
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import plotly.express as px
import scipy.stats as stats
from itertools import combinations

st.set_page_config(page_title="Dashboard Ejecutivo Estudiantil", layout="wide")

@st.cache_data(show_spinner=False)
def cargar_df_clean():
    path = Path('df_clean.csv')
    if not path.exists():
        raise FileNotFoundError("Se requiere df_clean.csv en el directorio actual.")
    df = pd.read_csv(path, index_col=0)
    return df.copy()

df = cargar_df_clean()

# Normalizar nombres clave para análisis (crear alias sin modificar originales)
rename_map = {
    'Curricular.units.1st.sem..enrolled.': 'enrolled_1',
    'Curricular.units.1st.sem..approved.': 'approved_1',
    'Curricular.units.1st.sem..evaluations.': 'evals_1',
    'Curricular.units.1st.sem..grade.': 'grade_1',
    'Curricular.units.2nd.sem..enrolled.': 'enrolled_2',
    'Curricular.units.2nd.sem..approved.': 'approved_2',
    'Curricular.units.2nd.sem..evaluations.': 'evals_2',
    'Curricular.units.2nd.sem..grade.': 'grade_2'
}
df_alias = df.rename(columns={k:v for k,v in rename_map.items() if k in df.columns}).copy()

#############################
# Funciones utilitarias core #
#############################

def safe_ratio(num, den):
    return np.where(den>0, num/den, np.nan)

def derivar_metricas_basicas(df_in: pd.DataFrame) -> pd.DataFrame:
    dfw = df_in.copy()
    # Ratios aprobación / densidad de evaluaciones
    deriv_pairs = [
        ('approved_1','enrolled_1','ratio_aprob_1'),
        ('approved_2','enrolled_2','ratio_aprob_2'),
        ('evals_1','enrolled_1','dens_eval_1'),
        ('evals_2','enrolled_2','dens_eval_2')
    ]
    for num, den, new_name in deriv_pairs:
        if num in dfw.columns and den in dfw.columns:
            dfw[new_name] = safe_ratio(dfw[num], dfw[den])
    if {'grade_1','grade_2'} <= set(dfw.columns):
        dfw['grade_mean'] = dfw[['grade_1','grade_2']].mean(axis=1)
        dfw['delta_grade'] = dfw['grade_2'] - dfw['grade_1']
    if {'ratio_aprob_1','ratio_aprob_2'} <= set(dfw.columns):
        dfw['delta_ratio_aprob'] = dfw['ratio_aprob_2'] - dfw['ratio_aprob_1']
    return dfw

df_alias = derivar_metricas_basicas(df_alias)

# Score administrativo simple (heurístico)
admin_sources = [c for c in df_alias.columns if any(k in c.lower() for k in ['tuition','debtor','scholar'])]
positivos = ['yes','up.to.date','paid','scholar']
def flag_favorable(val:str):
    s = str(val).lower()
    return int(any(p in s for p in positivos))
score_cols = []
for c in admin_sources:
    if df_alias[c].nunique() <= 12:
        cname = f'score_{c}'
        df_alias[cname] = df_alias[c].apply(flag_favorable)
        score_cols.append(cname)
if score_cols:
    df_alias['admin_score_sum'] = df_alias[score_cols].sum(axis=1)

TARGET_COL = 'Target'
if TARGET_COL not in df_alias.columns:
    st.error("No se encuentra columna 'Target' en df_clean.csv")
    st.stop()

# Mapeo de valores de Target a español (sin sobrescribir original para evitar romper lógica)
target_map = {'Graduate':'Graduado','Dropout':'Abandono','Enrolled':'Inscrito'}
df_alias['Target_es'] = df_alias[TARGET_COL].map(target_map).fillna(df_alias[TARGET_COL])
TARGET_DISPLAY = 'Target_es'

st.title("Dashboard Ejecutivo - Desempeño Estudiantil")
st.caption("Visión estratégica para la toma de decisiones (H1-H3)")
st.markdown(
    """
**Curso:** Ciencia de Datos y Big Data  
**Universidad:** Universidad Nacional de San Agustín de Arequipa  
**Autor:** Fredy Abel Huanca Torres
    """
)

# Sidebar filtros
with st.sidebar:
    st.header("Filtros")
    targets_sel = st.multiselect("Resultado (Target)", options=sorted(df_alias[TARGET_DISPLAY].dropna().unique()), default=sorted(df_alias[TARGET_DISPLAY].dropna().unique()))
    debtor_cols = [c for c in admin_sources if 'debtor' in c.lower()]
    tuition_cols = [c for c in admin_sources if 'tuition' in c.lower()]
    select_debtor = None
    if debtor_cols:
        dcol = debtor_cols[0]
        select_debtor = st.multiselect(dcol, options=sorted(df_alias[dcol].dropna().unique()), default=sorted(df_alias[dcol].dropna().unique()))
    select_tuition = None
    if tuition_cols:
        tcol = tuition_cols[0]
        select_tuition = st.multiselect(tcol, options=sorted(df_alias[tcol].dropna().unique()), default=sorted(df_alias[tcol].dropna().unique()))

filtro = df_alias[TARGET_DISPLAY].isin(targets_sel)
if debtor_cols and select_debtor is not None:
    filtro &= df_alias[debtor_cols[0]].isin(select_debtor)
if tuition_cols and select_tuition is not None:
    filtro &= df_alias[tuition_cols[0]].isin(select_tuition)

data = df_alias.loc[filtro].copy()

# KPIs principales
k1, k2, k3, k4 = st.columns(4)
k1.metric("Registros filtrados", len(data))
graduation_rate = data[TARGET_DISPLAY].value_counts(normalize=True).get('Graduado', 0)*100
dropout_rate = data[TARGET_DISPLAY].value_counts(normalize=True).get('Abandono', 0)*100
k2.metric("% Graduado", f"{graduation_rate:0.1f}%")
k3.metric("% Abandono", f"{dropout_rate:0.1f}%")
if 'grade_mean' in data.columns:
    k4.metric("Nota media global", f"{data['grade_mean'].mean():0.2f}")
else:
    k4.metric("Métrica disponible", "N/A")

tabs = st.tabs(["H1 - Desempeño temprano","H2 - Deterioro","H3 - Administrativo","Calidad de Datos","Conclusiones & Próximos Pasos"])

# Diccionario de etiquetas amigables para métricas
metric_labels = {
    'ratio_aprob_1':'Ratio aprobación Sem1',
    'ratio_aprob_2':'Ratio aprobación Sem2',
    'grade_1':'Nota Sem1',
    'grade_2':'Nota Sem2',
    'grade_mean':'Nota media',
    'approved_1':'Unidades aprobadas Sem1',
    'approved_2':'Unidades aprobadas Sem2',
    'enrolled_1':'Unidades inscritas Sem1',
    'enrolled_2':'Unidades inscritas Sem2',
    'delta_grade':'Δ Nota (Sem2 - Sem1)',
    'delta_ratio_aprob':'Δ Ratio aprobación',
    'admin_score_sum':'Score administrativo agregado'
}
def label(col):
    return metric_labels.get(col, col)

################################
# Funciones específicas H1 - H3 #
################################

def figs_h1(dfh: pd.DataFrame, target_col: str):
    cols = [c for c in ['ratio_aprob_1','grade_1','approved_1','enrolled_1'] if c in dfh.columns]
    figs = []
    for c in cols:
        serie = dfh[c].dropna()
        if serie.nunique() <= 5:  # barra si baja cardinalidad
            tmp = dfh.groupby(target_col)[c].mean().reset_index()
            f = px.bar(tmp, x=target_col, y=c, color=target_col, title=f"H1 – Distribución {label(c)} por estado", text=c)
            f.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        else:
            f = px.violin(dfh, x=target_col, y=c, box=True, points='all', color=target_col, title=f"H1 – Distribución {label(c)} por estado")
        f.update_layout(xaxis_title='', yaxis_title=label(c))
        figs.append(f)
    summary = None
    if cols:
        summary = dfh[cols + [target_col]].groupby(target_col).agg(['mean','median','std'])
    return figs, summary

def kruskal_eta(dfh, target_col, metric):
    grupos = [g[metric].dropna().values for _,g in dfh.groupby(target_col)]
    grupos_validos = [g for g in grupos if len(g)>1]
    if len(grupos_validos) < 2 or all(np.std(g)==0 for g in grupos_validos):
        return {'metrica':metric,'H':np.nan,'p':np.nan,'eta2':np.nan}
    H,p = stats.kruskal(*grupos_validos)
    N = sum(len(g) for g in grupos_validos)
    eta2 = H/(N-1) if N>1 else np.nan
    return {'metrica':metric,'H':H,'p':p,'eta2':eta2}

def dunn_posthoc(dfh, target_col, metric):
    datos = dfh[[target_col, metric]].dropna()
    if datos[target_col].nunique() < 3:
        return pd.DataFrame()
    datos['rank'] = datos[metric].rank()
    grupos = datos.groupby(target_col)
    ni = grupos.size(); ri = grupos['rank'].sum(); N = len(datos)
    if N < 6:
        return pd.DataFrame()
    rows = []
    for a,b in combinations(ni.index,2):
        n1,n2 = ni[a],ni[b]; r1,r2 = ri[a],ri[b]
        denom = np.sqrt((N*(N+1)/12) * (1/n1 + 1/n2))
        if denom == 0: continue
        z = (r1/n1 - r2/n2)/denom
        p = 2*(1 - stats.norm.cdf(abs(z)))
        rows.append({'metrica':metric,'grupo_a':a,'grupo_b':b,'z':z,'p_raw':p})
    if not rows:
        return pd.DataFrame()
    dfp = pd.DataFrame(rows).sort_values('p_raw').reset_index(drop=True)
    m = len(dfp)
    # BH
    dfp['p_adj_bh'] = dfp.apply(lambda r: min(dfp.loc[r.name:,'p_raw'].min()*m/(r.name+1),1), axis=1)
    return dfp

def h2_analysis(dfh: pd.DataFrame, target_col: str):
    # Reutiliza columnas derivadas ya calculadas en derivar_metricas_basicas
    work = dfh.copy()
    work = work[work[target_col].notna() & (work[target_col].astype(str).str.strip()!='') & (work[target_col].astype(str).str.lower()!='nan')].copy()
    metrics = [c for c in ['ratio_aprob_1','ratio_aprob_2','nota_sem1','nota_sem2','delta_ratio_aprob','delta_grade','delta_nota'] if c in work.columns]
    # Compatibilidad: en dashboard usamos grade_1/grade_2
    if 'grade_1' in work.columns and 'grade_2' in work.columns and 'delta_grade' not in work.columns:
        work['delta_grade'] = work['grade_2'] - work['grade_1']
        if 'nota_sem1' not in work.columns:
            work['nota_sem1'] = work['grade_1']
            work['nota_sem2'] = work['grade_2']
    if {'ratio_aprob_1','ratio_aprob_2'} <= set(work.columns) and 'delta_ratio_aprob' not in work.columns:
        work['delta_ratio_aprob'] = work['ratio_aprob_2'] - work['ratio_aprob_1']
    metrics = [m for m in ['ratio_aprob_1','ratio_aprob_2','nota_sem1','nota_sem2','delta_ratio_aprob','delta_grade'] if m in work.columns]
    figs = []
    stats_rows = []
    posthocs = []
    for m in metrics:
        serie = work[m].dropna()
        if len(serie) < 5 or serie.nunique() <= 1:
            continue
        # Figura
        if serie.nunique() <= 5:
            tmp = work.groupby(target_col)[m].mean().reset_index()
            f = px.bar(tmp, x=target_col, y=m, color=target_col, title=f"H2 – Distribución {m} por estado", text=m)
            f.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        else:
            f = px.violin(work, x=target_col, y=m, box=True, points='all', color=target_col, title=f"H2 – Distribución {m} por estado")
        f.update_layout(xaxis_title='', yaxis_title=m)
        figs.append(f)
        # Estadísticos
        krow = kruskal_eta(work, target_col, m)
        stats_rows.append(krow)
        if not np.isnan(krow['p']) and krow['p'] < 0.10:
            ph = dunn_posthoc(work, target_col, m)
            if not ph.empty:
                posthocs.append(ph)
    stats_df = pd.DataFrame(stats_rows)
    ph_df = pd.concat(posthocs, ignore_index=True) if posthocs else pd.DataFrame()
    # Resumen descriptivo
    resumen_list = []
    for m in metrics:
        if m not in work.columns: continue
        grp = work.groupby(target_col)[m].agg(['count','median','mean','std']).rename(columns={'count':'n'})
        q1 = work.groupby(target_col)[m].quantile(0.25)
        q3 = work.groupby(target_col)[m].quantile(0.75)
        grp['IQR'] = q3 - q1; grp['q1']=q1; grp['q3']=q3; grp['metrica']=m
        resumen_list.append(grp.reset_index())
    resumen_df = pd.concat(resumen_list, ignore_index=True) if resumen_list else pd.DataFrame()
    return {'figs':figs,'kruskal':stats_df,'posthoc':ph_df,'resumen':resumen_df}

def cramers_v_corrected(tab):
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.to_numpy().sum()
    r,k = tab.shape
    phi2 = chi2 / n if n>0 else np.nan
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1)) if n>1 else np.nan
    r_corr = r - ((r-1)**2)/(n-1) if n>1 else np.nan
    k_corr = k - ((k-1)**2)/(n-1) if n>1 else np.nan
    denom = min((k_corr-1),(r_corr-1)) if (r_corr is not None and k_corr is not None) else np.nan
    if denom and denom>0:
        return np.sqrt(phi2_corr / denom)
    return np.nan

def h3_analysis(dfh: pd.DataFrame, target_col: str):
    patrones_admin = ["fee","pago","tuition","administr","debt","mora","finan"]
    admin_candidates = [c for c in dfh.columns if any(p in c.lower() for p in patrones_admin)]
    valid = []
    omit = []
    for c in admin_candidates:
        nunq = dfh[c].nunique(dropna=True)
        if nunq <=1:
            omit.append({'variable':c,'motivo':'1 categoria'})
            continue
        if nunq > 8:
            omit.append({'variable':c,'motivo':'>8 categorias'})
            continue
        if dfh[c].isna().mean() > 0.40:
            omit.append({'variable':c,'motivo':'>40% NA'})
            continue
        valid.append(c)
    resultados = []
    figs = []
    for var in valid:
        sub = dfh[[target_col, var]].dropna().copy()
        if sub[var].nunique() < 2:
            omit.append({'variable':var,'motivo':'<2 categorias post-limpieza'})
            continue
        tab = pd.crosstab(sub[target_col], sub[var])
        if tab.shape[1] < 2:
            omit.append({'variable':var,'motivo':'<2 columnas en tabla'})
            continue
        try:
            chi2, p, dof, exp = stats.chi2_contingency(tab)
        except Exception as e:
            omit.append({'variable':var,'motivo':f'chi2 error: {e}'})
            continue
        cv = cramers_v_corrected(tab)
        resultados.append({'variable':var,'n_total':int(sub.shape[0]),'categorias':int(sub[var].nunique()),'chi2':chi2,'p_chi2':p,'dof':dof,'cramers_v':cv})
        freq = sub.groupby([target_col, var]).size().reset_index(name='conteo')
        total_estado = freq.groupby(target_col)['conteo'].transform('sum')
        freq['proporcion'] = freq['conteo']/total_estado
        titulo = f"H3 – Distribución {var} por estado (proporciones)"
        f = px.bar(freq, x=target_col, y='proporcion', color=var, title=titulo, text='proporcion')
        f.update_traces(texttemplate='%{text:.2%}', hovertemplate='%{x}<br>%{legendgroup}: %{y:.2%}<extra></extra>')
        f.update_layout(yaxis_tickformat='.0%', yaxis_title='Proporción', xaxis_title='')
        figs.append(f)
    return {
        'resultados': pd.DataFrame(resultados),
        'omitidas': pd.DataFrame(omit),
        'figs': figs
    }

# === H1 (nuevo) ===
with tabs[0]:
    st.subheader("H1: Desempeño académico temprano")
    st.markdown("Análisis de métricas del primer semestre y su asociación con el estado final.")
    figs1, summary1 = figs_h1(data, TARGET_DISPLAY)
    if not figs1:
        st.warning("No hay métricas detectadas para H1.")
    else:
        # Obtener nombres de métricas desde el eje Y (título correcto) en lugar de parsear el título de la figura
        metric_names = [f.layout.yaxis.title.text for f in figs1]
        sel = st.multiselect("Seleccionar métricas a mostrar", options=metric_names, default=metric_names)
        # Mostrar en el orden original filtrando por nombre exacto del eje Y
        for f in figs1:
            metric_name = f.layout.yaxis.title.text
            if metric_name in sel:
                st.plotly_chart(f, use_container_width=True)
        if summary1 is not None:
            st.markdown("**Resumen estadístico**")
            st.dataframe(summary1)

with tabs[1]:
    st.subheader("H2: Deterioro inter-semestre")
    st.markdown("Comparación de métricas entre primer y segundo semestre, deltas y contraste estadístico.")
    h2_res = h2_analysis(data, TARGET_DISPLAY)
    if not h2_res['figs']:
        st.info("No se generaron métricas válidas para H2.")
    else:
        for f in h2_res['figs']:
            st.plotly_chart(f, use_container_width=True)
        with st.expander("Resultados estadísticos (Kruskal / Dunn)"):
            st.markdown("**Kruskal-Wallis y eta² aproximado**")
            st.dataframe(h2_res['kruskal'])
            if not h2_res['posthoc'].empty:
                st.markdown("**Post-hoc Dunn (BH-FDR)**")
                st.dataframe(h2_res['posthoc'])
        with st.expander("Resumen descriptivo"):
            st.dataframe(h2_res['resumen'])
        # Flags de deterioro simples basadas en deltas presentes
        delta_cols_present = [c for c in ['delta_grade','delta_ratio_aprob'] if c in data.columns]
        if delta_cols_present:
            for dc in delta_cols_present:
                data[f'flag_deterioro_{dc}'] = (data[dc] < 0).astype(int)
            flags = [c for c in data.columns if c.startswith('flag_deterioro_')]
            st.markdown("**Proporción de estudiantes con deterioro (<0)**")
            st.dataframe(data.groupby(TARGET_DISPLAY)[flags].mean().T.style.format('{:.2%}'))

with tabs[2]:
    st.subheader("H3: Factores administrativos")
    st.markdown("Asociación entre variables administrativas y estado final (Chi² y Cramér's V corregido).")
    h3_res = h3_analysis(data, TARGET_DISPLAY)
    if not h3_res['figs']:
        st.info("No se detectaron variables administrativas válidas bajo los criterios definidos.")
    else:
        for f in h3_res['figs']:
            st.plotly_chart(f, use_container_width=True)
        with st.expander("Resultados Chi² / Cramér's V"):
            st.dataframe(h3_res['resultados'])
        if not h3_res['omitidas'].empty:
            with st.expander("Variables omitidas (motivo)"):
                st.dataframe(h3_res['omitidas'])
    if 'admin_score_sum' in data.columns:
        fig = px.box(data, x=TARGET_DISPLAY, y='admin_score_sum', color=TARGET_DISPLAY, title="H3 – Score administrativo agregado por estado")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Interpretación: Scores superiores en 'Graduado' refuerzan la relevancia administrativa.")

# === Calidad de Datos === (ajuste índice tras eliminar pestaña de pruebas)
with tabs[3]:
    st.subheader("Calidad de Datos: Sorpresas y Desafíos")
    st.markdown("""
    **Hallazgos de calidad:**
    - Flags de outliers creadas (prefijo `flag_out_`) incrementaron columnas (+12) sin eliminar datos: permiten monitorear casos extremos.
    - Variables potencialmente categóricas codificadas numéricamente (ocupación padres) requieren mapeo semántico para evitar ruido.
    - Ausencia de imputación avanzada: valores faltantes se mantienen (riesgo en modelado si no se manejan).
    - Posible colinealidad entre métricas derivadas (ratio_aprob y grade_mean) que necesita verificación (VIF / correlaciones) antes de modelos finales.
    - Deltas negativos recurrentes podrían ser señal temprana para intervención.
    """)
    # Tabla nulos
    nulls = df_alias.isna().sum()
    nulls_df = (nulls[nulls>0].sort_values(ascending=False).to_frame('nulos'))
    if not nulls_df.empty:
        st.markdown("**Columnas con nulos**")
        st.dataframe(nulls_df)
    # Outliers flags
    flag_cols = [c for c in df_alias.columns if c.startswith('flag_out_')]
    if flag_cols:
        preval = df_alias[flag_cols].mean().sort_values(ascending=False).to_frame('tasa').style.format('{:.2%}')
        st.markdown("**Prevalencia de flags de outliers**")
        st.dataframe(preval)
    # Cardinalidad
    cardinalidad = df_alias.nunique().sort_values(ascending=False).head(20).to_frame('cardinalidad_top')
    st.markdown("**Top 20 cardinalidad**")
    st.dataframe(cardinalidad)

# === Conclusiones === (ajuste índice)
with tabs[4]:
    st.subheader("Conclusiones Ejecutivas")
    st.markdown("""
    ### H1. Desempeño Académico Temprano
    **Conclusión:** Las métricas iniciales (ratio de aprobación y nota del primer semestre) exhiben diferencias sistemáticas entre estados finales: los graduados parten de una base más sólida, los inscritos intermedia y los abandonos rezagados. Esto respalda que el rendimiento temprano es un predictor crítico del desenlace.
    **Evidencia:** Distribuciones separadas y orden consistente (Graduado > Inscrito > Abandono) en aprobación y notas.
    **Implicación:** Activar alertas tras primer semestre para estudiantes bajo percentil 30.

    ### H2. Deterioro Inter-Semestre
    **Conclusión:** Los abandonos concentran mayor proporción de deltas negativos (caída en nota y/o ratio de aprobación); el deterioro es un indicador incremental de riesgo.
    **Evidencia:** Deltas (Δ nota, Δ ratio) con desplazamientos desfavorables y contrastes post-hoc en pares con Abandono.
    **Implicación:** Monitorear variaciones y activar tutorías cuando Δ < 0 combinado con base baja inicial.

    ### H3. Factores Administrativos
    **Conclusión:** Condiciones administrativas favorables (pagos al día, ausencia de mora, becas) se asocian con mayor graduación y menor abandono.
    **Evidencia:** χ² significativos y tamaños de efecto (Cramér's V) no triviales; proporciones favorables en categorías regulares.
    **Implicación:** Integrar señales administrativas en el modelo de riesgo y coordinar intervenciones financieras.

    ### Síntesis Integrada
    El desempeño temprano (H1) fija la línea base; el deterioro (H2) señala aceleración del riesgo; factores administrativos (H3) modulan la sostenibilidad académica. Se requiere combinar: (1) nivel inicial, (2) tendencia, (3) condición administrativa.

    ### Próximos Pasos Recomendados
    1. Modelo predictivo multietapa (logística + árbol) con features H1 + deltas H2 + flags administrativas.
    2. Definir umbrales operativos (prob. abandono > 0.65) y flujos de derivación.
    3. Validar estabilidad temporal con nuevas cohortes.
    4. Medir impacto de intervenciones sobre tasa de abandono en piloto controlado.
    5. Implementar pipeline de imputación y normalización para robustecer generalización.
    """)
    st.markdown("**Descarga dataset base (alias + derivadas)**")
    st.download_button(
        label="Descargar CSV",
        data=df_alias.to_csv(index=False).encode('utf-8'),
        file_name='df_clean_derivadas.csv',
        mime='text/csv'
    )

st.caption("© 2025 Dashboard Ejecutivo - Análisis Académico")
