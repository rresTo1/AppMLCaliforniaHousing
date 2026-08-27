
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title('Predicción del Valor de Viviendas en California')
st.write('Esta aplicación predice el valor mediano de una vivienda (en cientos de miles de dólares) basado en varias características, utilizando un modelo de regresión polinómica de grado 3.')

# Cargar los modelos serializados
@st.cache_resource
def load_models():
    scaler = joblib.load('minmax_scaler.pkl')
    poly_features_3 = joblib.load('poly_features_3.pkl')
    poly_model_3 = joblib.load('poly_model_3.pkl')
    return scaler, poly_features_3, poly_model_3

scaler, poly_features_3, poly_model_3 = load_models()

# Definir las características de entrada
features = ['MedInc', 'AveRooms', 'AveBedrms', 'AveOccup', 'Latitude']

# Crear la interfaz de usuario para la entrada de datos
st.sidebar.header('Introduce las características de la casa:')

def user_input_features():
    medinc = st.sidebar.slider('MedInc (Ingreso Mediano por Bloque)', 0.0, 15.0, 3.5)
    averooms = st.sidebar.slider('AveRooms (Promedio de Habitaciones)', 0.0, 15.0, 5.0)
    avebedrms = st.sidebar.slider('AveBedrms (Promedio de Dormitorios)', 0.0, 5.0, 1.0)
    aveoccup = st.sidebar.slider('AveOccup (Ocupación Promedio del Hogar)', 0.0, 10.0, 2.5)
    latitude = st.sidebar.slider('Latitude (Latitud)', 32.0, 42.0, 34.0)
    
    data = {
        'MedInc': medinc,
        'AveRooms': averooms,
        'AveBedrms': avebedrms,
        'AveOccup': aveoccup,
        'Latitude': latitude
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

st.subheader('Características de entrada especificadas:')
st.write(input_df)

# Realizar la predicción
if st.button('Predecir Costo'):
    # Escalar las características de entrada
    scaled_input = scaler.transform(input_df)
    scaled_input_df = pd.DataFrame(scaled_input, columns=input_df.columns)
    
    # Transformar a características polinómicas
    poly_input = poly_features_3.transform(scaled_input_df)
    
    # Predecir el valor
    prediction = poly_model_3.predict(poly_input)
    
    # El resultado se da en cientos de miles de dólares
    predicted_cost_dollars = prediction[0] * 100000
    
    st.subheader('Predicción del Valor Mediano de la Vivienda:')
    st.success(f'El costo predicho para la casa es: **${predicted_cost_dollars:,.2f}**')
    st.info('Nota: El valor predicho está en cientos de miles de dólares, la salida aquí ya ha sido convertida a dólares.')
