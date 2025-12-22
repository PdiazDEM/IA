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

# --- 1. CONFIGURACIÓN VISUAL (UX/UI) ---
st.set_page_config(page_title="MetAI Huechuraba", page_icon="⛈️", layout="centered")

# URL directa al logo
LOGO_URL = "https://www.huechuraba.cl/imagenes/logo_huechuraba.png"

st.markdown("""
    <style>
    /* 1. FONDO OSCURO CLEAN (Estilo GitHub Dark) */
    .stApp {
        background-color: #0d1117;
    }

    /* 2. FORZAR TEXTO BLANCO */
    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #e6edf3 !important;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* 3. ARREGLO DE NÚMEROS (MÉTRICAS) */
    div[data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="metric-container"] > div:nth-child(2) {
        color: #ffffff !important;
        font-weight: bold;
    }
    div[data-testid="metric-container"] > label {
        color: #8b949e !important;
    }

    /* 4. ESTILO DE LA TABLA (Para que se vea bien en oscuro) */
    [data-testid="stDataFrame"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 10px;
    }

    /* 5. BANNER AFTER SCHOOL */
    .after-school-box {
        background-color: #1f6feb;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        font-weight: 500;
        border: 1px solid #388bfd;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.3);
    }

    /* 6. BOTÓN PRINCIPAL */
    .stButton>button {
        width: 100%;
        background-color: #238636;
        color: white !important;
        font-weight: bold;
        border: none;
        border-radius: 6px;
        padding: 12px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        transform: scale(1.02);
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER CON LOGO Y TÍTULO ---
col_head1, col_head2 = st.columns([1, 4])

with col_head1:
    try:
        st.image(LOGO_URL, width=100)
    except:
        st.write("🏛️")

with col_head2:
    st.markdown("# MetAI Huechuraba")
    st.markdown("##### Inteligencia Artificial Meteorológica")

# --- 3. MENCIÓN PROGRAMA AFTER SCHOOL ---
st.markdown("""
<div class="after-school-box">
    🚀 <strong>Programa After School - Municipalidad de Huechuraba</strong><br>
    <span style="font-size: 0.9em; opacity: 0.9;">
    Proyecto de Ciencia de Datos desarrollado por estudiantes para la comunidad.
    </span>
</div>
""", unsafe_allow_html=True)

# --- 4. CARGA DE MODELOS ---
@st.cache_resource
def load_models():
    try:
        modelo = joblib.load('modelo_lluvia_v3.joblib')
        scaler = joblib.load('scaler_lluvia_v3.joblib')
        return modelo, scaler
    except: return None, None

modelo, scaler = load_models()

if modelo is None:
    st.error("⚠️ Error de sistema: No se encontraron los modelos IA.")
    st.stop()

# --- 5. LÓGICA DE DATOS (Backend) ---
def get_history():
    try:
        data = Daily('85574', datetime.now() - timedelta(days=15), datetime.now())
        df = data.fetch()
        if 'pres' not in df.columns: df['pres'] = 1013.25
        df = df[['tavg', 'tmin', 'tmax', 'prcp', 'pres']].ffill()
        df = df.fillna({'pres': 1013.25, 'prcp': 0})
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

# --- 6. INTERFAZ DE USUARIO ---

col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    btn_calc = st.button("ANALIZAR CLIMA AHORA")

if btn_calc:
    with st.spinner("📡 Conectando con satélites y ejecutando modelo..."):
        memoria = get_history()
        futuro = get_forecast()
        
        if memoria is None or futuro is None:
            st.warning("⚠️ Sin conexión a datos meteorológicos. Intenta más tarde.")
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
        
        # --- RESULTADOS ---
        st.markdown("---")
        hoy = df.iloc[0]
        st.markdown(f"### 📅 Pronóstico: {hoy['Fecha'].strftime('%A %d')}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡️ Máxima", f"{hoy['Max']:.1f}°C")
        c2.metric("💧 Riesgo Lluvia", f"{hoy['Prob']:.1%}")
        
        with c3:
            st.write("")
            if hoy['Prob'] > 0.5: st.error("☔ LLUVIA")
            elif hoy['Prob'] > 0.2: st.warning("☁️ POSIBLE")
            else: st.success("☀️ SECO")

        # GRÁFICO
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Tendencia Semanal")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Fecha'], y=df['Prob'], mode='lines+markers', name='Probabilidad',
            fill='tozeroy', line=dict(color='#2f81f7', width=4), marker=dict(size=6, color='white')
        ))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=20), height=300,
            yaxis=dict(tickformat='.0%', range=[0, 1], gridcolor='#30363d'),
            xaxis=dict(gridcolor='#30363d', tickformat='%a %d')
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- TABLA DE DATOS DETALLADA (AÑADIDA) ---
        st.markdown("---")
        st.markdown("#### 📋 Detalle de Datos")
        
        # Preparar tabla para visualización limpia
        df_show = df.copy()
        df_show['Fecha'] = df_show['Fecha'].dt.strftime('%d-%m-%Y')
        df_show['Prob'] = df_show['Prob'].apply(lambda x: f"{x:.1%}")
        df_show['Max'] = df_show['Max'].apply(lambda x: f"{x:.1f} °C")
        df_show.columns = ['Fecha', 'Probabilidad Lluvia', 'Temp. Máxima'] # Renombrar para que se vea bonito
        
        st.dataframe(df_show, use_container_width=True, hide_index=True)

else:
    st.info("👋 Pulsa el botón verde para descargar datos satelitales.")
