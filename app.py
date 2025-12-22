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

# --- 1. CONFIGURACIÓN DE PÁGINA Y ESTILO CSS ---
st.set_page_config(page_title="Santiago Weather IA", page_icon="⛈️", layout="centered")

# CSS Personalizado para dar look "App Moderna"
st.markdown("""
    <style>
    /* Fondo degradado suave */
    .stApp {
        background: linear-gradient(to bottom right, #1e3c72, #2a5298);
        color: white;
    }
    /* Estilo de las métricas */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    /* Títulos */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Helvetica Neue', sans-serif;
    }
    /* Botón principal */
    .stButton>button {
        width: 100%;
        background-color: #ffcc00;
        color: #000000;
        font-weight: bold;
        border-radius: 20px;
        border: none;
        padding: 10px 20px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #ffd633;
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Encabezado
st.title("⛈️ Santiago Weather AI")
st.markdown("### Pronóstico Inteligente Híbrido (v3.0)")
st.caption("Powered by XGBoost & Open-Meteo • Precisión basada en Presión Atmosférica")

# --- 2. CARGAR MODELOS (LÓGICA ORIGINAL) ---
@st.cache_resource
def load_models():
    try:
        modelo = joblib.load('modelo_lluvia_v3.joblib')
        scaler = joblib.load('scaler_lluvia_v3.joblib')
        return modelo, scaler
    except Exception as e:
        return None, None

modelo, scaler = load_models()

if modelo is None:
    st.error("🚨 Error Crítico: No se encuentran los archivos .joblib")
    st.info("Por favor sube 'modelo_lluvia_v3.joblib' y 'scaler_lluvia_v3.joblib'")
    st.stop()

# --- 3. FUNCIONES DE DATOS (LÓGICA ORIGINAL) ---
def get_history():
    station_id = '85574' # Pudahuel
    end = datetime.now()
    start = end - timedelta(days=15)
    
    try:
        data = Daily(station_id, start, end)
        df = data.fetch()
        if 'pres' not in df.columns: df['pres'] = 1013.25
        df = df[['tavg', 'tmin', 'tmax', 'prcp', 'pres']].ffill()
        df['pres'] = df['pres'].fillna(1013.25)
        df['prcp'] = df['prcp'].fillna(0)
        df['lloviendo_hoy'] = (df['prcp'] > 0.1).astype(int)
        df = df.rename(columns={'prcp': 'prcp_hoy'})
        if len(df) < 3: return None
        return df.tail(3)
    except:
        return None

def get_forecast():
    try:
        cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": -33.3833, "longitude": -70.7833,
            "daily": ["temperature_2m_max", "temperature_2m_min", "pressure_msl_mean"],
            "timezone": "auto", "forecast_days": 8
        }
        
        responses = openmeteo.weather_api(url, params=params)
        daily = responses[0].Daily()
        
        # Procesamiento
        daily_tmax = daily.Variables(0).ValuesAsNumpy()
        daily_tmin = daily.Variables(1).ValuesAsNumpy()
        daily_pres = daily.Variables(2).ValuesAsNumpy()
        daily_tavg = (daily_tmax + daily_tmin) / 2
        
        dates = pd.date_range(
            start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
            end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
            freq = pd.Timedelta(seconds = daily.Interval()), inclusive = "left"
        )
        
        return {'tmax': daily_tmax, 'tmin': daily_tmin, 'tavg': daily_tavg, 'pres': daily_pres, 'fechas': dates}
    except:
        return None

# --- 4. INTERFAZ Y PREDICCIÓN ---

col_btn1, col_btn2, col_btn3 = st.columns([1,2,1])
with col_btn2:
    run_btn = st.button("🔍 ANALIZAR CLIMA AHORA")

if run_btn:
    with st.spinner("📡 Consultando satélites y ejecutando red neuronal..."):
        
        memoria_df = get_history()
        pronostico = get_forecast()
        
        if memoria_df is None or pronostico is None:
            st.error("⚠️ Error de conexión con los servicios meteorológicos.")
            st.stop()

        # --- Lógica de Predicción (Tu cerebro IA) ---
        cols_base = ['tavg', 'tmin', 'tmax', 'prcp_hoy', 'lloviendo_hoy', 'pres']
        features = [ # LISTA EXACTA V3.0
            'tavg', 'tmin', 'tmax', 'pres', 
            'dia_año_sen', 'dia_año_cos', 'dia_sem_sen', 'dia_sem_cos', 
            'tavg_lag_1', 'tavg_lag_2', 'tavg_lag_3', 
            'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3', 
            'tmax_lag_1', 'tmax_lag_2', 'tmax_lag_3', 
            'prcp_hoy_lag_1', 'prcp_hoy_lag_2', 'prcp_hoy_lag_3', 
            'lloviendo_hoy_lag_1', 'lloviendo_hoy_lag_2', 'lloviendo_hoy_lag_3', 
            'pres_lag_1', 'pres_lag_2', 'pres_lag_3'
        ]
        
        predicciones = []
        
        for i in range(1, 8):
            datos_input = pd.DataFrame(index=[0])
            datos_input['tavg'] = pronostico['tavg'][i]
            datos_input['tmin'] = pronostico['tmin'][i]
            datos_input['tmax'] = pronostico['tmax'][i]
            datos_input['pres'] = pronostico['pres'][i]
            datos_input['prcp_hoy'] = 0.0; datos_input['lloviendo_hoy'] = 0
            
            for col in cols_base:
                datos_input[f'{col}_lag_1'] = memoria_df.iloc[-1][col]
                datos_input[f'{col}_lag_2'] = memoria_df.iloc[-2][col]
                datos_input[f'{col}_lag_3'] = memoria_df.iloc[-3][col]
            
            fecha = pronostico['fechas'][i]
            da = fecha.dayofyear; ds = fecha.dayofweek
            datos_input['dia_año_sen'] = np.sin(2*np.pi*da/366)
            datos_input['dia_año_cos'] = np.cos(2*np.pi*da/366)
            datos_input['dia_sem_sen'] = np.sin(2*np.pi*ds/7)
            datos_input['dia_sem_cos'] = np.cos(2*np.pi*ds/7)
            
            for f in features:
                if f not in datos_input.columns: datos_input[f] = 0
            
            datos_input = datos_input[features]
            datos_scaled = scaler.transform(datos_input)
            prob = modelo.predict_proba(datos_scaled)[0][1]
            
            predicciones.append({'Fecha': fecha, 'Probabilidad': prob, 'T_Max': pronostico['tmax'][i]})
            
            nueva_fila = pd.DataFrame([{
                'tavg': pronostico['tavg'][i], 'tmin': pronostico['tmin'][i],
                'tmax': pronostico['tmax'][i], 'pres': pronostico['pres'][i],
                'prcp_hoy': 0, 'lloviendo_hoy': 1 if prob > 0.4 else 0
            }])
            memoria_df = pd.concat([memoria_df.iloc[1:], nueva_fila], ignore_index=True)

        # --- MOSTRAR RESULTADOS PREMIUM ---
        df_res = pd.DataFrame(predicciones)
        
        # 1. Tarjeta Principal (Mañana)
        mañana = df_res.iloc[0]
        prob_mañana = mañana['Probabilidad']
        fecha_mañana = mañana['Fecha'].strftime("%A %d")
        
        st.markdown("---")
        st.markdown(f"<h2 style='text-align: center;'>Pronóstico para Mañana: {fecha_mañana}</h2>", unsafe_allow_html=True)
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        col_res1.metric("🌡️ Máxima", f"{mañana['T_Max']:.1f}°C")
        col_res2.metric("💧 Prob. Lluvia", f"{prob_mañana:.1%}")
        
        with col_res3:
            if prob_mañana > 0.5:
                st.error("☔ LLEVA PARAGUAS")
            elif prob_mañana > 0.2:
                st.warning("☁️ NUBLADO/RIESGO")
            else:
                st.success("😎 DÍA SECO")

        # 2. Gráfico Interactivo (Plotly)
        st.markdown("### 📅 Tendencia a 7 Días")
        
        # Crear gráfico moderno
        fig = go.Figure()
        
        # Área de probabilidad
        fig.add_trace(go.Scatter(
            x=df_res['Fecha'], y=df_res['Probabilidad'],
            fill='tozeroy', mode='lines+markers',
            name='Prob. Lluvia',
            line=dict(color='#00CC96', width=3),
            marker=dict(size=8)
        ))
        
        # Línea de temperatura (eje secundario visual)
        fig.add_trace(go.Scatter(
            x=df_res['Fecha'], y=df_res['T_Max']/40, # Normalizado visualmente
            mode='lines', name='Temp (Ref)',
            line=dict(color='#EF553B', width=2, dash='dot')
        ))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickformat='.0%'),
            margin=dict(l=0, r=0, t=30, b=0),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # 3. Tabla Detallada
        with st.expander("Ver tabla de datos detallada"):
            st.table(df_res.style.format({'Probabilidad': '{:.1%}', 'T_Max': '{:.1f}°C'}))

else:
    # Mensaje de bienvenida
    st.markdown("""
    <div style='text-align: center; padding: 50px; opacity: 0.7;'>
        <h3>👈 Pulsa el botón para activar la IA</h3>
        <p>El sistema descargará los últimos datos de presión atmosférica y calculará el riesgo real.</p>
    </div>
    """, unsafe_allow_html=True)
