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

# --- 1. CONFIGURACIÓN VISUAL (ESTILO CLARO / LIGHT MODE) ---
st.set_page_config(page_title="MetAI Huechuraba", page_icon="⛈️", layout="centered")

# URL de tu logo nuevo
LOGO_URL = "https://i.imgur.com/HjqQolt.png"

st.markdown("""
    <style>
    /* 1. FONDO GENERAL (Gris Suave Profesional) */
    .stApp {
        background-color: #F0F2F6;
        color: #1F2937;
    }

    /* 2. TIPOGRAFÍA Y TEXTOS */
    h1, h2, h3, h4 {
        color: #111827 !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    p, label, div {
        color: #374151;
    }

    /* 3. CONTENEDORES / TARJETAS (Efecto "Card" Blanco) */
    div[data-testid="metric-container"] {
        background-color: #FFFFFF; /* Fondo blanco */
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    div[data-testid="metric-container"] > div:nth-child(2) {
        color: #1F2937 !important; /* Número oscuro fuerte */
        font-weight: 800;
    }
    div[data-testid="metric-container"] > label {
        color: #6B7280 !important; /* Etiqueta gris medio */
    }

    /* 4. BANNER AFTER SCHOOL (Estilo Institucional) */
    .after-school-box {
        background: linear-gradient(135deg, #2563EB 0%, #1E40AF 100%); /* Azul degradado */
        color: white !important;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 25px;
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.3);
    }
    .after-school-box strong {
        color: white !important;
        font-size: 1.1em;
    }
    .after-school-box span {
        color: #DBEAFE !important;
    }

    /* 5. BOTÓN PRINCIPAL */
    .stButton>button {
        width: 100%;
        background-color: #10B981; /* Verde esmeralda */
        color: white !important;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 14px;
        font-size: 16px;
        transition: all 0.2s;
        box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
    }
    .stButton>button:hover {
        background-color: #059669;
        transform: translateY(-2px);
    }

    /* 6. TABLA DE DATOS (Estilo limpio) */
    [data-testid="stDataFrame"] {
        background-color: white;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
    }
    
    /* 7. AJUSTES DEL LOGO */
    .logo-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        max-width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. HEADER CON TU LOGO ---
col_logo, col_text = st.columns([1, 3])

with col_logo:
    # Usamos HTML para controlar mejor el tamaño y centrado del logo
    st.markdown(f'<img src="{LOGO_URL}" class="logo-img" width="130">', unsafe_allow_html=True)

with col_text:
    st.markdown("<div style='padding-top: 10px;'>", unsafe_allow_html=True)
    st.markdown("# MetAI Huechuraba")
    st.markdown("##### Inteligencia Artificial Meteorológica")
    st.markdown("</div>", unsafe_allow_html=True)

# --- 3. MENCIÓN PROGRAMA AFTER SCHOOL ---
st.markdown("""
<div class="after-school-box">
    🚀 <strong>Programa After School - Municipalidad de Huechuraba</strong><br>
    <span>Innovación y Ciencia de Datos al servicio de la comunidad.</span>
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
    st.error("⚠️ Error: No se encuentran los archivos del modelo IA.")
    st.stop()

# --- 5. LÓGICA DE DATOS (Backend) ---
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

# --- 6. INTERFAZ ---

col_b1, col_b2, col_b3 = st.columns([1, 2, 1])
with col_b2:
    btn_calc = st.button("ANALIZAR CLIMA AHORA")

if btn_calc:
    with st.spinner("📡 Procesando datos satelitales..."):
        memoria = get_history()
        futuro = get_forecast()
        
        if memoria is None or futuro is None:
            st.warning("⚠️ Sin conexión a datos externos.")
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
        
        # --- DISPLAY RESULTADOS ---
        st.markdown("---")
        hoy = df.iloc[0]
        st.markdown(f"### 📅 Pronóstico: {hoy['Fecha'].strftime('%A %d')}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡️ Máxima", f"{hoy['Max']:.1f}°C")
        c2.metric("💧 Riesgo Lluvia", f"{hoy['Prob']:.1%}")
        
        with c3:
            st.write("")
            if hoy['Prob'] > 0.5: st.error("☔ ALTO RIESGO")
            elif hoy['Prob'] > 0.2: st.warning("☁️ POSIBLE")
            else: st.success("☀️ SECO")

        # GRÁFICO (Estilo Light)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Tendencia Semanal")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Fecha'], y=df['Prob'], mode='lines+markers', name='Riesgo',
            fill='tozeroy', 
            line=dict(color='#2563EB', width=3), # Azul Institucional
            marker=dict(size=6, color='white', line=dict(width=2, color='#2563EB'))
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=30, b=20), height=300,
            yaxis=dict(tickformat='.0%', range=[0, 1], gridcolor='#E5E7EB'), # Rejilla gris suave
            xaxis=dict(gridcolor='#E5E7EB', tickformat='%a %d'),
            font=dict(color='#374151') # Texto del gráfico gris oscuro
        )
        st.plotly_chart(fig, use_container_width=True)

        # TABLA DE DATOS
        st.markdown("---")
        st.markdown("#### 📋 Detalle de Datos")
        
        df_show = df.copy()
        df_show['Fecha'] = df_show['Fecha'].dt.strftime('%d-%m-%Y')
        df_show['Prob'] = df_show['Prob'].apply(lambda x: f"{x:.1%}")
        df_show['Max'] = df_show['Max'].apply(lambda x: f"{x:.1f} °C")
        df_show.columns = ['Fecha', 'Probabilidad Lluvia', 'Temp. Máxima']
        
        st.dataframe(df_show, use_container_width=True, hide_index=True)

else:
    st.info("👋 Pulsa el botón verde para iniciar el análisis.")
