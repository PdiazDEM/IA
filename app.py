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

# --- 1. CONFIGURACIÓN Y ESTILO (EL ARREGLO VISUAL) ---
st.set_page_config(page_title="Santiago Weather IA", page_icon="⛈️", layout="centered")

st.markdown("""
    <style>
    /* Fondo Degradado Profundo (Azul Noche) */
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        background-attachment: fixed;
    }
    
    /* Forzar texto blanco en títulos y párrafos */
    h1, h2, h3, p, label, .stMarkdown {
        color: #f0f2f6 !important;
    }

    /* Estilo para las Tarjetas de Métricas (Glassmorphism) */
    div[data-testid="metric-container"] {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="metric-container"] > label {
        color: #a0a0a0 !important; /* Etiqueta gris claro */
    }
    div[data-testid="metric-container"] > div {
        color: #ffffff !important; /* Valor blanco brillante */
    }

    /* Botón Principal */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #ff8c00 0%, #ff0080 100%);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-size: 16px;
        transition: transform 0.2s;
        box-shadow: 0 4px 15px rgba(255, 0, 128, 0.4);
    }
    .stButton>button:hover {
        transform: scale(1.03);
        color: white;
    }

    /* Arreglo para la Tabla (Fondo blanco suave para leer datos) */
    .stTable {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 10px;
        color: black !important;
    }
    /* Forzar color negro dentro de la tabla si es necesario */
    .stTable td, .stTable th {
        color: #333333 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("⛈️ Santiago Weather AI")
st.markdown("##### 🤖 Predicción Híbrida: XGBoost + Presión Atmosférica")

# --- 2. CARGAR MODELOS ---
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
    st.error("🚨 Error: Faltan archivos .joblib. Súbelos a GitHub.")
    st.stop()

# --- 3. FUNCIONES DE DATOS ---
def get_history():
    station_id = '85574'
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
    except: return None

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
    except: return None

# --- 4. LÓGICA Y VISUALIZACIÓN ---

if st.button("🔮 Calcular Pronóstico de Riesgo"):
    
    with st.spinner("Conectando con satélites..."):
        memoria_df = get_history()
        pronostico = get_forecast()
        
        if memoria_df is None or pronostico is None:
            st.warning("⚠️ Sin conexión a datos externos. Revisa la API.")
            st.stop()

        # Preparar Features (Exactas V3.0)
        features = [
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

        df_res = pd.DataFrame(predicciones)
        
        # --- DISEÑO DE RESULTADOS ---
        mañana = df_res.iloc[0]
        prob_val = mañana['Probabilidad']
        
        st.markdown("---")
        st.markdown(f"### 🗓️ Pronóstico: {mañana['Fecha'].strftime('%A %d')}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🌡️ Máxima", f"{mañana['T_Max']:.1f}°C")
        col2.metric("💧 Riesgo Lluvia", f"{prob_val:.1%}")
        
        with col3:
            st.write("") # Espaciador
            if prob_val > 0.5:
                st.error("☔ **ALTA PROBABILIDAD**")
            elif prob_val > 0.2:
                st.warning("☁️ **POSIBLE LLUVIA**")
            else:
                st.success("☀️ **DÍA SECO**")
        
        # --- GRÁFICO CORREGIDO (MODO OSCURO) ---
        st.markdown("### 📈 Tendencia Semanal")
        
        fig = go.Figure()
        
        # Línea de Probabilidad (Área con degradado)
        fig.add_trace(go.Scatter(
            x=df_res['Fecha'], y=df_res['Probabilidad'],
            mode='lines+markers',
            name='Probabilidad',
            fill='tozeroy',
            line=dict(color='#00d2ff', width=3), # Cian brillante
            marker=dict(size=8, color='white', line=dict(width=2, color='#00d2ff'))
        ))
        
        # Línea de Referencia de Temperatura
        fig.add_trace(go.Scatter(
            x=df_res['Fecha'], y=df_res['T_Max']/40, # Escalada para visualización
            mode='lines',
            name='Temp (Ref)',
            line=dict(color='#ff9900', width=2, dash='dot')
        ))

        fig.update_layout(
            template='plotly_dark', # <--- ESTO ARREGLA EL FONDO NEGRO/TEXTO
            paper_bgcolor='rgba(0,0,0,0)', # Fondo transparente para integrarse
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=30, b=10),
            height=350,
            yaxis=dict(tickformat='.0%', title="Probabilidad", gridcolor='rgba(255,255,255,0.1)'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            legend=dict(orientation="h", y=1.1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # --- TABLA CORREGIDA ---
        st.markdown("### 📋 Detalle de Datos")
        # Usamos st.dataframe en lugar de table para mejor manejo de temas, 
        # pero el CSS de arriba forzará que se vea claro.
        df_display = df_res.copy()
        df_display['Probabilidad'] = df_display['Probabilidad'].apply(lambda x: f"{x:.1%}")
        df_display['T_Max'] = df_display['T_Max'].apply(lambda x: f"{x:.1f}°C")
        df_display['Fecha'] = df_display['Fecha'].dt.strftime('%Y-%m-%d')
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)

else:
    # Pantalla de Inicio Limpia
    st.info("👋 Pulsa el botón 'Calcular' para iniciar el análisis IA.")
