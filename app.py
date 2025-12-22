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

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILO UX/UI ---
st.set_page_config(page_title="MetAI Huechuraba", page_icon="🌧️", layout="centered")

# URL del Logo de Huechuraba (Puedes cambiar esto por un archivo local si lo subes a GitHub)
LOGO_URL = "https://www.huechuraba.cl/img/logo-huechuraba.png" 

st.markdown("""
    <style>
    /* IMPORTAR FUENTE MODERNA (Roboto/Inter style) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* RESET GENERAL */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* FONDO MINIMALISTA OSCURO */
    .stApp {
        background-color: #0E1117; /* Gris muy oscuro, casi negro (GitHub Dark theme style) */
        color: #E6E6E6;
    }

    /* HEADER Y TÍTULOS */
    h1 {
        font-weight: 800;
        letter-spacing: -1px;
        color: #FFFFFF;
        margin-bottom: 0px;
    }
    h3 {
        font-weight: 400;
        color: #A0A0A0;
        font-size: 1.1rem;
    }
    .description-box {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        font-size: 0.95rem;
        line-height: 1.5;
        color: #C9D1D9;
    }

    /* TARJETAS DE MÉTRICAS (Card UI) */
    div[data-testid="metric-container"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        border-color: #58A6FF; /* Azul al pasar el mouse */
    }
    div[data-testid="metric-container"] > label {
        color: #8B949E !important;
        font-size: 0.8rem;
    }
    div[data-testid="metric-container"] > div {
        color: #F0F6FC !important;
    }

    /* BOTÓN DE ACCIÓN (Call to Action) */
    .stButton>button {
        width: 100%;
        background-color: #238636; /* Verde GitHub/Huechuraba */
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 20px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2EA043;
        box-shadow: 0 4px 12px rgba(46, 160, 67, 0.4);
    }

    /* FOOTER */
    .footer {
        text-align: center;
        margin-top: 50px;
        padding-top: 20px;
        border-top: 1px solid #30363D;
        color: #8B949E;
        font-size: 0.8rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER CON LOGO (Diseño 2 Columnas) ---
col_logo, col_title = st.columns([1, 4])

with col_logo:
    # Mostramos el logo. Si falla la URL, no rompe la app.
    try:
        st.image(LOGO_URL, width=100)
    except:
        st.write("🏛️") # Fallback icon

with col_title:
    st.markdown("# MetAI Huechuraba")
    st.markdown("### Inteligencia Artificial Meteorológica")

# --- 3. DESCRIPCIÓN DEL PROYECTO (Contexto Educativo) ---
st.markdown("""
<div class="description-box">
    <strong>🏫 Proyecto Educativo After School</strong><br>
    Esta aplicación fue desarrollada por estudiantes del taller de programación de la 
    <strong>Municipalidad de Huechuraba</strong>. <br><br>
    Utiliza un modelo de Inteligencia Artificial (XGBoost) que analiza datos históricos de nuestra comuna 
    y pronósticos de presión atmosférica para predecir lluvias con mayor precisión local.
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
    st.error("⚠️ Archivos del modelo no encontrados. Por favor verifica el repositorio.")
    st.stop()

# --- 5. FUNCIONES DE DATOS (Backend) ---
def get_history():
    # Estación Pudahuel (Referencia más cercana fiable)
    try:
        data = Daily('85574', datetime.now() - timedelta(days=15), datetime.now())
        df = data.fetch()
        if 'pres' not in df.columns: df['pres'] = 1013.25
        df = df[['tavg', 'tmin', 'tmax', 'prcp', 'pres']].ffill().fillna({'pres':1013.25, 'prcp':0})
        df['lloviendo_hoy'] = (df['prcp'] > 0.1).astype(int)
        df = df.rename(columns={'prcp': 'prcp_hoy'})
        return df.tail(3) if len(df) >= 3 else None
    except: return None

def get_forecast():
    try:
        s = requests_cache.CachedSession('.cache', expire_after=3600)
        r = retry(s, retries=5, backoff_factor=0.2)
        om = openmeteo_requests.Client(session=r)
        
        # Coordenadas Huechuraba aprox
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

col_space_1, col_btn, col_space_2 = st.columns([1, 2, 1])

with col_btn:
    calcular = st.button("CALCULAR PROBABILIDAD DE LLUVIA")

if calcular:
    with st.spinner("🔄 Procesando datos atmosféricos..."):
        memoria = get_history()
        futuro = get_forecast()
        
        if memoria is None or futuro is None:
            st.error("Error conectando con los servidores de clima. Intenta más tarde.")
            st.stop()

        # FEATURES V3.0
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
        
        res = []
        cols_base = ['tavg', 'tmin', 'tmax', 'prcp_hoy', 'lloviendo_hoy', 'pres']

        for i in range(1, 8):
            row = pd.DataFrame(index=[0])
            # Datos actuales
            row['tavg'] = futuro['tavg'][i]; row['tmin'] = futuro['tmin'][i]
            row['tmax'] = futuro['tmax'][i]; row['pres'] = futuro['pres'][i]
            row['prcp_hoy'] = 0.0; row['lloviendo_hoy'] = 0
            
            # Lags y Ciclos
            for c in cols_base:
                row[f'{c}_lag_1'] = memoria.iloc[-1][c]
                row[f'{c}_lag_2'] = memoria.iloc[-2][c]
                row[f'{c}_lag_3'] = memoria.iloc[-3][c]
                
            f_date = futuro['fechas'][i]
            dy = f_date.dayofyear; dw = f_date.dayofweek
            row['dia_año_sen'] = np.sin(2*np.pi*dy/366); row['dia_año_cos'] = np.cos(2*np.pi*dy/366)
            row['dia_sem_sen'] = np.sin(2*np.pi*dw/7); row['dia_sem_cos'] = np.cos(2*np.pi*dw/7)
            
            # Ordenar
            for f in features_list: 
                if f not in row.columns: row[f] = 0
            
            # Predecir
            prob = modelo.predict_proba(scaler.transform(row[features_list]))[0][1]
            res.append({'Fecha': f_date, 'Prob': prob, 'Max': futuro['tmax'][i]})
            
            # Update Memoria
            new_row = pd.DataFrame([{'tavg': futuro['tavg'][i], 'tmin': futuro['tmin'][i], 
                                     'tmax': futuro['tmax'][i], 'pres': futuro['pres'][i], 
                                     'prcp_hoy': 0, 'lloviendo_hoy': 1 if prob > 0.4 else 0}])
            memoria = pd.concat([memoria.iloc[1:], new_row], ignore_index=True)

        df = pd.DataFrame(res)
        
        # --- DISPLAY RESULTADOS (Diseño Minimalista) ---
        st.markdown("---")
        
        hoy = df.iloc[0]
        st.markdown(f"### 📅 Pronóstico para mañana: {hoy['Fecha'].strftime('%A %d')}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Temperatura Máx", f"{hoy['Max']:.1f}°C")
        c2.metric("Prob. Lluvia", f"{hoy['Prob']:.1%}")
        
        with c3:
            st.write("")
            if hoy['Prob'] > 0.5:
                st.markdown("🚨 <span style='color:#FF6B6B; font-weight:bold'>LLUVIA PROBABLE</span>", unsafe_allow_html=True)
            elif hoy['Prob'] > 0.2:
                st.markdown("☁️ <span style='color:#FFD93D; font-weight:bold'>POSIBLE LLUVIA</span>", unsafe_allow_html=True)
            else:
                st.markdown("☀️ <span style='color:#4DAB9A; font-weight:bold'>DÍA SECO</span>", unsafe_allow_html=True)

        # Gráfico Minimalista
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Fecha'], y=df['Prob'], mode='lines', name='Riesgo',
                                 line=dict(color='#58A6FF', width=4), fill='tozeroy'))
        fig.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=20, b=20), height=250,
            yaxis=dict(showgrid=True, gridcolor='#30363D', tickformat='.0%'),
            xaxis=dict(showgrid=False)
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    # Estado inicial vacío (Limpio)
    pass

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Municipalidad de Huechuraba • Programa After School<br>
    Desarrollado con ❤️ y Python
</div>
""", unsafe_allow_html=True)
