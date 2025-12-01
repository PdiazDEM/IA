import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from meteostat import Daily, Stations
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- 1. CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Predicctor de Lluvia Santiago", page_icon="🌧️")

st.title("🌧️ IA Meteorológica: Santiago")
st.write("Predicción de lluvia basada en XGBoost Híbrido (Histórico + Pronóstico)")

# --- 2. CARGAR MODELOS ---
@st.cache_resource
def load_models():
    modelo = joblib.load('modelo_lluvia_v3.joblib')
    scaler = joblib.load('scaler_lluvia_v3.joblib')
    return modelo, scaler

try:
    modelo, scaler = load_models()
    st.success("✅ Cerebro IA cargado correctamente.")
except:
    st.error("❌ No se encontraron los archivos .joblib. Asegúrate de subirlos.")
    st.stop()

# --- 3. OBTENER HISTORIA RECIENTE (METEOSTAT) ---
# Necesitamos los últimos 3 días reales para la "memoria" del modelo
def get_history():
    # Estación Pudahuel
    station_id = '85574' 
    end = datetime.now()
    start = end - timedelta(days=10) # Pedimos 10 días por seguridad
    
    data = Daily(station_id, start, end)
    df = data.fetch()
    
    # Preparación idéntica al entrenamiento
    if 'pres' not in df.columns: df['pres'] = 1013.25
    df = df[['tavg', 'tmin', 'tmax', 'prcp', 'pres']].ffill()
    df['pres'] = df['pres'].fillna(1013.25)
    df['prcp'] = df['prcp'].fillna(0)
    df['lloviendo_hoy'] = (df['prcp'] > 0.1).astype(int)
    df = df.rename(columns={'prcp': 'prcp_hoy'})
    
    return df.tail(3) # Devolvemos solo los últimos 3 días

# --- 4. OBTENER PRONÓSTICO FUTURO (OPEN-METEO) ---
def get_forecast():
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": -33.3833, 
        "longitude": -70.7833,
        "daily": ["temperature_2m_max", "temperature_2m_min", "pressure_msl_mean"],
        "timezone": "auto",
        "forecast_days": 8
    }
    
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    daily = response.Daily()
    
    daily_tmax = daily.Variables(0).ValuesAsNumpy()
    daily_tmin = daily.Variables(1).ValuesAsNumpy()
    daily_pres = daily.Variables(2).ValuesAsNumpy()
    daily_tavg = (daily_tmax + daily_tmin) / 2
    
    dates = pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )
    
    return {
        'tmax': daily_tmax, 'tmin': daily_tmin, 'tavg': daily_tavg,
        'pres': daily_pres, 'fechas': dates
    }

# --- 5. BOTÓN DE PREDICCIÓN ---
if st.button("🔮 Calcular Probabilidad de Lluvia"):
    
    with st.spinner("Consultando satélites y ejecutando modelo neuronal..."):
        
        # A. Obtener datos
        try:
            memoria_df = get_history()
            pronostico = get_forecast()
        except Exception as e:
            st.error(f"Error al obtener datos externos: {e}")
            st.stop()

        # B. Bucle de Predicción (Tu lógica maestra)
        cols_base = ['tavg', 'tmin', 'tmax', 'prcp_hoy', 'lloviendo_hoy', 'pres']
        features = ['lag_1', 'lag_2', 'lag_3', 'media_movil_7d', # OJO: Si usaste media movil, añade la lógica, si no, quítala. 
                    # Asumiré las features de tu última versión v3.0:
                    'tavg_lag_1', 'tavg_lag_2', 'tavg_lag_3',
                    'tmin_lag_1', 'tmin_lag_2', 'tmin_lag_3',
                    'tmax_lag_1', 'tmax_lag_2', 'tmax_lag_3',
                    'prcp_hoy_lag_1', 'prcp_hoy_lag_2', 'prcp_hoy_lag_3',
                    'lloviendo_hoy_lag_1', 'lloviendo_hoy_lag_2', 'lloviendo_hoy_lag_3',
                    'pres_lag_1', 'pres_lag_2', 'pres_lag_3',
                    'dia_año_sen', 'dia_año_cos', 'dia_sem_sen', 'dia_sem_cos']
        
        # NOTA: Ajusta la lista 'features' ARRIBA para que sea IDÉNTICA 
        # a la que usaste en la Celda 7 de tu entrenamiento.
        
        predicciones = []
        
        for i in range(1, 8):
            datos_input = pd.DataFrame(index=[0])
            
            # Datos del día
            datos_input['tavg'] = pronostico['tavg'][i]
            datos_input['tmin'] = pronostico['tmin'][i]
            datos_input['tmax'] = pronostico['tmax'][i]
            datos_input['pres'] = pronostico['pres'][i]
            datos_input['prcp_hoy'] = 0.0; datos_input['lloviendo_hoy'] = 0
            
            # Lags
            for col in cols_base:
                datos_input[f'{col}_lag_1'] = memoria_df.iloc[-1][col]
                datos_input[f'{col}_lag_2'] = memoria_df.iloc[-2][col]
                datos_input[f'{col}_lag_3'] = memoria_df.iloc[-3][col]
            
            # Cíclicos
            fecha = pronostico['fechas'][i]
            da = fecha.dayofyear; ds = fecha.dayofweek
            datos_input['dia_año_sen'] = np.sin(2*np.pi*da/366)
            datos_input['dia_año_cos'] = np.cos(2*np.pi*da/366)
            datos_input['dia_sem_sen'] = np.sin(2*np.pi*ds/7)
            datos_input['dia_sem_cos'] = np.cos(2*np.pi*ds/7)
            
            # Relleno y orden
            # Importante: El scaler espera las columnas en orden exacto
            # Aquí hacemos un truco para asegurar que todas existan
            for f in features:
                if f not in datos_input.columns: datos_input[f] = 0
            datos_input = datos_input[features]
            
            # Predecir
            datos_scaled = scaler.transform(datos_input)
            prob = modelo.predict_proba(datos_scaled)[0][1]
            
            predicciones.append({'Fecha': fecha, 'Probabilidad': prob, 'T_Max': pronostico['tmax'][i]})
            
            # Actualizar memoria
            nueva_fila = pd.DataFrame([{
                'tavg': pronostico['tavg'][i], 'tmin': pronostico['tmin'][i],
                'tmax': pronostico['tmax'][i], 'pres': pronostico['pres'][i],
                'prcp_hoy': 0, 'lloviendo_hoy': 1 if prob > 0.4 else 0
            }])
            memoria_df = pd.concat([memoria_df.iloc[1:], nueva_fila], ignore_index=True)

    # --- 6. MOSTRAR RESULTADOS ---
    df_res = pd.DataFrame(predicciones)
    df_res['Fecha'] = df_res['Fecha'].dt.strftime('%A %d')
    
    # Métricas principales (Mañana)
    st.subheader(f"Pronóstico para mañana: {df_res.iloc[0]['Fecha']}")
    
    col1, col2, col3 = st.columns(3)
    prob_mañana = df_res.iloc[0]['Probabilidad']
    
    col1.metric("Probabilidad Lluvia", f"{prob_mañana:.1%}", delta_color="inverse")
    col2.metric("Temp. Máxima", f"{df_res.iloc[0]['T_Max']:.1f}°C")
    
    if prob_mañana > 0.5:
        col3.error("🌧️ Se espera Lluvia")
    elif prob_mañana > 0.2:
        col3.warning("☁️ Posible Lluvia")
    else:
        col3.success("☀️ Día Seco")

    # Gráfico
    st.subheader("Tendencia 7 Días")
    st.bar_chart(data=df_res.set_index('Fecha')['Probabilidad'])
    
    # Tabla
    st.table(df_res.style.format({'Probabilidad': '{:.2%}', 'T_Max': '{:.1f}°C'}))