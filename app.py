import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from meteostat import Daily
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="AI MET Huechuraba", page_icon="📡", layout="wide")

# URL del Logo
LOGO_URL = "https://i.imgur.com/HjqQolt.png"

# --- 2. ESTILOS CSS PERSONALIZADOS (MODERNO / DOCENTE) ---
st.markdown("""
    <style>
    /* Importar fuente moderna */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
    }

    /* ESTILO BARRA LATERAL */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #dee2e6;
    }

    /* ESTILO HEADER PRINCIPAL */
    .main-header {
        background: linear-gradient(90deg, #0052cc 0%, #00a3e0 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 82, 204, 0.2);
    }
    .main-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2.2rem;
    }
    .main-header h3 {
        color: #e0e7ff !important;
        margin: 0;
        font-weight: 300;
        font-size: 1.1rem;
    }

    /* TARJETAS DE RESULTADOS */
    div[data-testid="metric-container"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #0052cc;
    }

    /* BOTÓN DE ACCIÓN */
    .stButton>button {
        width: 100%;
        background-color: #10B981;
        color: white;
        border-radius: 8px;
        font-weight: bold;
        padding: 0.75rem;
        border: none;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.2);
    }
    .stButton>button:hover {
        background-color: #059669;
    }
    
    /* CAJA DE CÓDIGO (Librerías) */
    .lib-box {
        font-family: 'Courier New', monospace;
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 10px;
        border-radius: 5px;
        font-size: 0.85rem;
        border-left: 4px solid #0052cc;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BARRA LATERAL (INFORMACIÓN Y LIBRERÍAS) ---
with st.sidebar:
    st.image(LOGO_URL, use_container_width=True)
    
    st.markdown("### 🏫 Programa After School")
    st.info("""
    **Proyecto de Innovación Educativa**
    Municipalidad de Huechuraba.
    
    Este software fue diseñado por estudiantes para aplicar conceptos de **Ciencia de Datos** en problemas reales de la comuna.
    """)
    
    st.markdown("---")
    st.markdown("### 🛠️ Librerías Utilizadas")
    st.markdown("""
    Este proyecto integra las tecnologías más potentes de Python para análisis de datos:
    """)
    
    # Lista de librerías con estilo de código
    st.markdown("""
    <div class="lib-box">
    import streamlit as web<br>
    import pandas as data<br>
    import sklearn as ai<br>
    import xgboost as model<br>
    import plotly as charts<br>
    import openmeteo as sat
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("v2.4.0 • Build for Education")

# --- 4. PANEL PRINCIPAL ---

# Header Personalizado HTML
st.markdown("""
<div class="main-header">
    <h1>AI MET Huechuraba</h1>
    <h3>Sistema Inteligente de Predicción Meteorológica Local</h3>
</div>
""", unsafe_allow_html=True)

# Descripción del funcionamiento (Explicación técnica simplificada)
with st.expander("📘 ¿Cómo funciona este modelo?", expanded=True):
    st.markdown("""
    Este sistema no es un pronóstico tradicional. Es un modelo híbrido que combina:
    1.  **Datos Históricos:** 15 años de comportamiento climático en la zona norte de Santiago.
    2.  **Modelo XGBoost:** Un algoritmo de Inteligencia Artificial que detecta patrones complejos entre la Presión Atmosférica y la Temperatura.
    3.  **Datos Satelitales:** Conexión en tiempo real con la API *Open-Meteo* para obtener las condiciones actuales.
    """)

# --- 5. CARGA DE MODELOS ---
@st.cache_resource
def load_models():
    try:
        modelo = joblib.load('modelo_lluvia_v3.joblib')
        scaler = joblib.load('scaler_lluvia_v3.joblib')
        return modelo, scaler
    except: return None, None

modelo, scaler = load_models()

if modelo is None:
    st.error("🚨 Error Crítico: Modelos IA no encontrados en el servidor.")
    st.stop()

# --- 6. LOGICA (Backend) ---
def get_history():
    try:
        data = Daily('85574', datetime.now() - timedelta(days=15), datetime.now())
        df = data.fetch()
        if 'pres' not in df.columns: df['pres'] = 1013.25
        df = df[['tavg', 'tmin', 'tmax', 'prcp', 'pres']].ffill().fillna({'pres': 1013.25, 'prcp': 0})
        df['lloviendo_hoy'] = (df['prcp'] > 0.1).astype(int)
        df = df.rename(columns={'prcp': 'prcp_hoy'})
        return df.tail(3) if len(df) >= 3 else None
    except: return None

def get_forecast():
    try:
        s = requests_cache.CachedSession('.cache', expire_after=3600)
        r = retry(s, retries=5, backoff_factor=0.2)
        om = openmeteo_requests.Client(session=r)
        params = {"latitude": -33.37, "longitude": -70.64, 
                  "daily": ["temperature_2m_max", "temperature_2m_min", "pressure_msl_mean"],
                  "timezone": "auto", "forecast_days": 8}
        res = om.weather_api("https://api.open-meteo.com/v1/forecast", params=params)[0]
        daily = res.Daily()
        d = {
            'tmax': daily.Variables(0).ValuesAsNumpy(),
            'tmin': daily.Variables(1).ValuesAsNumpy(),
            'tavg': (daily.Variables(0).ValuesAsNumpy() + daily.Variables(1).ValuesAsNumpy())/2,
            'pres': daily.Variables(2).ValuesAsNumpy(),
            'fechas': pd.date_range(start=pd.to_datetime(daily.Time(), unit="s", utc=True), 
                                    end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True), 
                                    freq=pd.Timedelta(seconds=daily.Interval()), inclusive="left")
        }
        return d
    except: return None

# --- 7. INTERFAZ Y RESULTADOS ---

# Botón central
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    btn_run = st.button("🚀 EJECUTAR ANÁLISIS IA")

if btn_run:
    with st.spinner("🔄 Descargando telemetría y procesando vectores..."):
        memoria = get_history()
        futuro = get_forecast()
        
        if memoria is None or futuro is None:
            st.warning("⚠️ Error de conexión API.")
            st.stop()

        features_list = [
            'tavg', 'tmin', 'tmax', 'pres', 
            'dia_año_sen', 'dia_año_cos', 'dia_sem_sen', 'dia_sem_cos', 
            'tavg_lag_1', 'tavg_lag_2', 'tavg_lag_3', 
            'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3', 
            'tmax_lag_1', 'tmax_lag_2', 'tmax_lag_3', 
            'prcp_hoy_lag_1', 'prcp_hoy_lag_2', 'prcp_hoy_lag_3', 
            'lloviendo_hoy_lag_1', 'lloviendo_hoy_lag_2', 'lloviendo_hoy_lag_3', 
            'pres_lag_1', 'pres_lag_2', 'pres_lag_3'
        ]
        cols_base = ['tavg', 'tmin', 'tmax', 'prcp_hoy', 'lloviendo_hoy', 'pres']
        res = []

        for i in range(1, 8):
            row = pd.DataFrame(index=[0])
            row['tavg'] = futuro['tavg'][i]; row['tmin'] = futuro['tmin'][i]
            row['tmax'] = futuro['tmax'][i]; row['pres'] = futuro['pres'][i]
            row['prcp_hoy'] = 0.0; row['lloviendo_hoy'] = 0
            
            for c in cols_base:
                row[f'{c}_lag_1'] = memoria.iloc[-1][c]
                row[f'{c}_lag_2'] = memoria.iloc[-2][c]
                row[f'{c}_lag_3'] = memoria.iloc[-3][c]
                
            dt = futuro['fechas'][i]
            dy = dt.dayofyear; dw = dt.dayofweek
            row['dia_año_sen'] = np.sin(2*np.pi*dy/366); row['dia_año_cos'] = np.cos(2*np.pi*dy/366)
            row['dia_sem_sen'] = np.sin(2*np.pi*dw/7); row['dia_sem_cos'] = np.cos(2*np.pi*dw/7)
            
            for f in features_list: 
                if f not in row.columns: row[f] = 0
            
            prob = modelo.predict_proba(scaler.transform(row[features_list]))[0][1]
            res.append({'Fecha': dt, 'Prob': prob, 'Max': futuro['tmax'][i]})
            
            new_row = pd.DataFrame([{'tavg': futuro['tavg'][i], 'tmin': futuro['tmin'][i], 
                                     'tmax': futuro['tmax'][i], 'pres': futuro['pres'][i], 
                                     'prcp_hoy': 0, 'lloviendo_hoy': 1 if prob > 0.4 else 0}])
            memoria = pd.concat([memoria.iloc[1:], new_row], ignore_index=True)

        df = pd.DataFrame(res)
        
        # --- VISUALIZACIÓN ---
        st.markdown("### 📊 Resultados del Modelo")
        hoy = df.iloc[0]
        
        # Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡️ Temp. Máxima", f"{hoy['Max']:.1f}°C")
        c2.metric("💧 Probabilidad Lluvia", f"{hoy['Prob']:.1%}", delta_color="inverse")
        
        with c3:
            st.write("")
            if hoy['Prob'] > 0.5: 
                st.markdown("<div style='background:#fee2e2; color:#991b1b; padding:10px; border-radius:5px; text-align:center; font-weight:bold;'>☔ ALERTA DE LLUVIA</div>", unsafe_allow_html=True)
            elif hoy['Prob'] > 0.2: 
                st.markdown("<div style='background:#fef3c7; color:#92400e; padding:10px; border-radius:5px; text-align:center; font-weight:bold;'>☁️ NUBLADO / RIESGO</div>", unsafe_allow_html=True)
            else: 
                st.markdown("<div style='background:#d1fae5; color:#065f46; padding:10px; border-radius:5px; text-align:center; font-weight:bold;'>☀️ DÍA DESPEJADO</div>", unsafe_allow_html=True)

        # Gráfico
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Fecha'], y=df['Prob'], mode='lines+markers', name='Probabilidad',
            fill='tozeroy', line=dict(color='#0052cc', width=3), marker=dict(size=7, color='white', line=dict(width=2, color='#0052cc'))
        ))
        fig.update_layout(
            title="Tendencia de Riesgo (Próximos 7 días)",
            template='plotly_white',
            margin=dict(l=20, r=20, t=40, b=20), height=350,
            yaxis=dict(tickformat='.0%', range=[0, 1], title="Probabilidad (0-100%)"),
            xaxis=dict(tickformat='%a %d')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tabla
        st.markdown("### 📋 Datos Detallados")
        df_show = df.copy()
        df_show['Fecha'] = df_show['Fecha'].dt.strftime('%d-%m-%Y')
        df_show['Prob'] = df_show['Prob'].apply(lambda x: f"{x:.1%}")
        df_show['Max'] = df_show['Max'].apply(lambda x: f"{x:.1f} °C")
        df_show.columns = ['Fecha', 'Probabilidad Lluvia', 'Temp. Máxima']
        st.dataframe(df_show, use_container_width=True, hide_index=True)

else:
    st.info("👈 Revisa el panel lateral para más información sobre el proyecto.")


