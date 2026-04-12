import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go

# =========================================
# 1. CONFIGURACIÓN DE LA PÁGINA Y CABECERA
# =========================================
st.set_page_config(page_title="Mantenimiento Predictivo - IMPALA", layout="wide")

import streamlit.components.v1 as components

# Ajustamos ligeramente las proporciones de las columnas para mejor distribución
col_logo, col_titulo, col_reloj = st.columns([1.2, 6.5, 2])

with col_logo:
    try:
        # Añadimos un pequeño margen superior invisible para que el logo baje y se alinee con el texto
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.image("logo_americorp.png", width="stretch")
    except FileNotFoundError:
        st.write("Logo no encontrado")

with col_titulo:
    # Diseño de título corporativo usando HTML/CSS puro
    html_titulo = """
    <div style="padding-top: 8px; padding-left: 15px;">
        <h1 style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 32px; font-weight: 800; color: var(--text-color); margin-bottom: 0px; line-height: 1.2;">
            ⚙️ Dashboard de Confiabilidad
        </h1>
        <h2 style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 22px; font-weight: 400; color: #00d2ff; margin-top: 5px; margin-bottom: 8px;">
            Faja Transportadora Impala
        </h2>
        <p style="font-family: monospace; font-size: 12px; color: #64748B; letter-spacing: 1px; margin-top: 0px;">
            MONITOREO PREDICTIVO DEL SOBREESFUERZO MECÁNICO MEDIANTE MACHINE LEARNING
        </p>
    </div>
    """
    st.markdown(html_titulo, unsafe_allow_html=True)

with col_reloj:
    # El reloj en vivo se mantiene intacto
    codigo_reloj_js = """
    <div id="reloj_container" style="text-align: right; padding-top: 15px; font-family: sans-serif;">
        <div id="hora" style="font-size: 26px; font-weight: bold; font-family: monospace; letter-spacing: 2px;"></div>
        <div id="fecha" style="font-size: 14px; margin-top: 2px;"></div>
    </div>

    <script>
        function actualizarReloj() {
            const ahora = new Date();
            
            const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
            document.getElementById('hora').style.color = isDark ? '#E2E8F0' : '#0F172A';
            document.getElementById('fecha').style.color = isDark ? '#94A3B8' : '#475569';

            let horas = ahora.getHours();
            let minutos = ahora.getMinutes().toString().padStart(2, '0');
            let segundos = ahora.getSeconds().toString().padStart(2, '0');
            let ampm = horas >= 12 ? 'pm' : 'am';
            
            horas = horas % 12;
            horas = horas ? horas : 12; 
            let horasStr = horas.toString().padStart(2, '0');
            
            let opcionesFecha = { weekday: 'long', year: 'numeric', month: 'long', day: '2-digit' };
            let fechaStr = ahora.toLocaleDateString('es-ES', opcionesFecha);
            fechaStr = fechaStr.charAt(0).toUpperCase() + fechaStr.slice(1); 
            
            document.getElementById('hora').innerText = `${horasStr}:${minutos}:${segundos} ${ampm}`;
            document.getElementById('fecha').innerText = fechaStr;
        }
        
        setInterval(actualizarReloj, 1000);
        actualizarReloj(); 
    </script>
    """
    components.html(codigo_reloj_js, height=90)

# ==========================================
# 2. CARGA DE MODELOS (CACHÉ PARA VELOCIDAD)
# ==========================================
@st.cache_resource
def cargar_modelos():
    modelo = joblib.load('isolation_forest_faja.joblib')
    scaler = joblib.load('scaler_faja.joblib')
    columnas = joblib.load('columnas_modelo.joblib')
    return modelo, scaler, columnas

try:
    modelo_if, scaler, variables_requeridas = cargar_modelos()
    st.sidebar.success("✅ IA Cargada Correctamente")
except Exception as e:
    st.sidebar.error("❌ Error al cargar los archivos .joblib. Verifica que estén en la misma carpeta.")
    st.stop()

# =========================================
# 3. CARGA DE DATOS (SUBIDA MANUAL)
# =========================================
st.sidebar.header("1. Carga de Datos")
archivo_subido = st.sidebar.file_uploader("📂 Sube el historial CSV del SCADA", type=["csv"])

# Si no hay archivo, mostramos un mensaje de bienvenida y detenemos la ejecución
if archivo_subido is None:
    st.info("👋 ¡Hola! Sube un archivo CSV de la Faja Impala en el panel izquierdo para comenzar el análisis.")
    st.stop()

# Si hay archivo, lo leemos
try:
    df = pd.read_csv(archivo_subido)
except Exception as e:
    st.error(f"❌ Error al leer el archivo CSV: {e}")
    st.stop()

# Aseguramos que el índice temporal exista
if 'timestampseconds' in df.columns:
    df['Fecha_Hora'] = pd.to_datetime(df['timestampseconds'])
    df.set_index('Fecha_Hora', inplace=True)
else:
    st.error("❌ El archivo subido no tiene la columna 'timestampseconds'.")
    st.stop()

with st.spinner("Procesando variables crudas y ejecutando IA..."):
    # 3.1 Feature Engineering (Ratio)
    if 'Ratio_Esfuerzo' not in df.columns:
        df['Ratio_Esfuerzo'] = df['Corriente_Salida_A'] / (df['Frecuencia_Hz'] + 0.001)
        
    # 3.2 Detección de Faja Apagada (Corriente casi cero)
    df['Estado'] = np.where(df['Corriente_Salida_A'] < 0.5, 'Apagada', 'Operando')
    
    # 3.3 Predicción del Modelo (Aquí descartamos las columnas crudas que no sirven)
    try:
        df_modelo = df[variables_requeridas].copy()
    except KeyError as e:
        st.error(f"❌ Faltan columnas vitales en el CSV subido. El modelo necesita: {variables_requeridas}")
        st.stop()
        
    datos_escalados = scaler.transform(df_modelo)
    df['Prediccion_Cruda'] = modelo_if.predict(datos_escalados)
    
    # 3.4 Regla de Negocio (Filtro < 80A)
    umbral_corriente = 80
    df['Alarma_Modelo'] = df['Prediccion_Cruda']
    df.loc[(df['Prediccion_Cruda'] == -1) & (df['Corriente_Salida_A'] < umbral_corriente), 'Alarma_Modelo'] = 1

# ==========================================
# 4. FILTROS DE FECHA Y HORA INTERACTIVOS
# ==========================================
st.sidebar.header("2. Filtros de Análisis")

# 4.1 Selección de Fecha de Inicio y Fin
fecha_inicio = st.sidebar.date_input("Fecha Inicio", df.index.min().date())
fecha_fin = st.sidebar.date_input("Fecha Fin", df.index.max().date())

# 4.2 Selección de Hora y Minuto (Usamos slider para que sea intuitivo)
st.sidebar.markdown("---")
st.sidebar.write("Rango Horario:")
hora_inicio = st.sidebar.time_input("Hora Inicio", value=pd.to_datetime('00:00').time())
hora_fin = st.sidebar.time_input("Hora Fin", value=pd.to_datetime('23:59').time())

# 4.3 Combinamos Fechas y Horas para crear el filtro final
import datetime
# Creamos variables datetime completas
datetime_inicio = datetime.datetime.combine(fecha_inicio, hora_inicio)
datetime_fin = datetime.datetime.combine(fecha_fin, hora_fin)

# 4.4 Aplicamos el filtro al DataFrame
mascara = (df.index >= datetime_inicio) & (df.index <= datetime_fin)
df_filtrado = df.loc[mascara]

if df_filtrado.empty:
    st.warning("No hay datos para este rango de fecha y hora.")
    st.stop()
# ==========================================
# 5. CÁLCULOS DE MÉTRICAS (EXCLUYENDO APAGADAS)
# ==========================================
df_operando = df_filtrado[df_filtrado['Estado'] == 'Operando']

# 5.1 Conteo de estados (Basado en la física de la Corriente)
total_monitoreo = len(df_operando)
datos_normales = len(df_operando[df_operando['Corriente_Salida_A'] < 180])
datos_precaucion = len(df_operando[(df_operando['Corriente_Salida_A'] >= 180) & (df_operando['Corriente_Salida_A'] < 200)])
datos_criticos = len(df_operando[df_operando['Corriente_Salida_A'] >= 200])

# 5.2 Conteo de IA y Fallas Finales
alertas_ia = len(df_operando[df_operando['Prediccion_Cruda'] == -1]) # Lo que dice el modelo crudo
anomalias_confirmadas = len(df_operando[df_operando['Alarma_Modelo'] == -1]) # IA validada con la regla de >80A

# 5.3 Riesgo y Acción
porcentaje_riesgo = (anomalias_confirmadas / total_monitoreo * 100) if total_monitoreo > 0 else 0

if porcentaje_riesgo < 5.0:
    estado_riesgo, color_estado, accion = "Normal", "normal", "🟢 OPERACIÓN NOMINAL: Continuar monitoreo"
elif porcentaje_riesgo <= 10.0:
    estado_riesgo, color_estado, accion = "Precaución", "off", "🟡 PRECAUCIÓN: Programar inspección visual"
else:
    estado_riesgo, color_estado, accion = "Crítico", "inverse", "🔴 ESTADO CRÍTICO: Priorizar intervención mecánica"

# ==========================================
# 6. PANEL DE CONTROL (KPI DASHBOARD PROFESIONAL)
# ==========================================

# 6.1 Barra Principal Estilo SCADA (Diseño replicado)
html_kpi_bar = f"""
<div style="display: flex; justify-content: space-between; background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; padding: 15px; margin-bottom: 20px;">
    <div style="flex: 1; border-right: 1px solid rgba(128,128,128,0.2); padding-left: 10px;">
        <div style="font-size: 11px; color: #64748B; font-family: monospace; letter-spacing: 1px;">MONITOREO TOTAL</div>
        <div style="font-size: 28px; font-weight: bold; color: #00d2ff; font-family: monospace;">{total_monitoreo}</div>
    </div>
    <div style="flex: 1; border-right: 1px solid rgba(128,128,128,0.2); padding-left: 20px;">
        <div style="font-size: 11px; color: #64748B; font-family: monospace; letter-spacing: 1px;">ESTADO NORMAL</div>
        <div style="font-size: 28px; font-weight: bold; color: #10B981; font-family: monospace;">{datos_normales}</div>
    </div>
    <div style="flex: 1; border-right: 1px solid rgba(128,128,128,0.2); padding-left: 20px;">
        <div style="font-size: 11px; color: #64748B; font-family: monospace; letter-spacing: 1px;">PRECAUCIÓN</div>
        <div style="font-size: 28px; font-weight: bold; color: #FBBF24; font-family: monospace;">{datos_precaucion}</div>
    </div>
    <div style="flex: 1; border-right: 1px solid rgba(128,128,128,0.2); padding-left: 20px;">
        <div style="font-size: 11px; color: #64748B; font-family: monospace; letter-spacing: 1px;">CRÍTICO</div>
        <div style="font-size: 28px; font-weight: bold; color: #EF4444; font-family: monospace;">{datos_criticos}</div>
    </div>
    <div style="flex: 1; padding-left: 20px;">
        <div style="font-size: 11px; color: #64748B; font-family: monospace; letter-spacing: 1px;">ALERTAS (IA)</div>
        <div style="font-size: 28px; font-weight: bold; color: #A855F7; font-family: monospace;">{alertas_ia}</div>
    </div>
</div>
"""
st.markdown(html_kpi_bar, unsafe_allow_html=True)

# 6.2 Fila Secundaria: Conclusión Predictiva y Acción
st.markdown("<p style='font-family: monospace; color: #64748B; font-size: 12px; margin-bottom: 10px; letter-spacing: 1px;'>DIAGNÓSTICO DEL SISTEMA</p>", unsafe_allow_html=True)

col_r1, col_r2, col_r3 = st.columns([1, 1, 2])
col_r1.metric("Anomalías Confirmadas", f"{anomalias_confirmadas}")
col_r2.metric("Riesgo de Falla", f"{porcentaje_riesgo:.2f}%", delta=estado_riesgo, delta_color=color_estado)

# Usamos success, warning o error dependiendo de la gravedad para pintar el cuadro de acción
if porcentaje_riesgo < 5.0:
    col_r3.success(f"**Acción de Mantenimiento:**\n{accion}")
elif porcentaje_riesgo <= 10.0:
    col_r3.warning(f"**Acción de Mantenimiento:**\n{accion}")
else:
    col_r3.error(f"**Acción de Mantenimiento:**\n{accion}")
# ==========================================
# 7. GRÁFICA INTERACTIVA PLOTLY (Scattergl para rapidez)
# ==========================================
st.markdown("---")
st.subheader("📈 Evolución de la Condición Mecánica")
st.subheader("Ratio de Esfuerzo v.s. Tiempo")

rango_tiempo = [df_filtrado.index.min(), df_filtrado.index.max()]

df_normal = df_filtrado[(df_filtrado['Alarma_Modelo'] == 1) & (df_filtrado['Estado'] == 'Operando')]
df_falla = df_filtrado[(df_filtrado['Alarma_Modelo'] == -1) & (df_filtrado['Estado'] == 'Operando')]
df_apagada = df_filtrado[df_filtrado['Estado'] == 'Apagada']

fig = go.Figure()

# Operación Normal (Azul)
fig.add_trace(go.Scattergl(x=df_normal.index, y=df_normal['Ratio_Esfuerzo'],
                         mode='markers', name='Operación Normal', 
                         marker=dict(color='#1f77b4', size=4, opacity=0.5)))

# Faja Apagada (Gris)
fig.add_trace(go.Scattergl(x=df_apagada.index, y=df_apagada['Ratio_Esfuerzo'],
                         mode='markers', name='Faja Apagada', 
                         marker=dict(color='grey', size=3, opacity=0.3)))

# Anomalía Confirmada (Rojo)
fig.add_trace(go.Scattergl(x=df_falla.index, y=df_falla['Ratio_Esfuerzo'],
                         mode='markers', name='Alerta de Sobreesfuerzo', 
                         marker=dict(color='red', size=8, symbol='x')))

fig.update_layout(
    xaxis_title="Fecha y Hora", 
    yaxis_title="Ratio de Esfuerzo (Corriente / Hz)", 
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(range=rango_tiempo) 
)

st.plotly_chart(fig, width="stretch", key="grafica_ratio")

# ==========================================
# 8. GRÁFICA DE CORRIENTE VS TIEMPO CON ZONAS OPERATIVAS
# ==========================================
st.markdown("---")
st.subheader("Corriente v.s Tiempo")

fig2 = go.Figure()

fig2.add_trace(go.Scattergl(x=df_filtrado.index, y=df_filtrado['Corriente_Salida_A'],
                         mode='lines', name='Corriente Real (A)',
                         line=dict(color='#00d2ff', width=1.5)))

# Línea Verde: Carga Nominal
fig2.add_hline(y=150, line_dash="dash", line_color="#00cc96", line_width=2,
               annotation_text="OPERACIÓN NOMINAL (150A)", 
               annotation_position="bottom right", annotation_font_color="#00cc96")

# Línea Amarilla: Alerta de Esfuerzo
fig2.add_hline(y=180, line_dash="dash", line_color="#ffa15a", line_width=2,
               annotation_text="ZONA DE ALERTA / ALTA CARGA (180A)", 
               annotation_position="bottom right", annotation_font_color="#ffa15a")

# Línea Roja: Límite Crítico
fig2.add_hline(y=200, line_dash="dash", line_color="#ef553b", line_width=2,
               annotation_text="ZONA CRÍTICA / SOBREESFUERZO (>200A)", 
               annotation_position="top right", annotation_font_color="#ef553b")

fig2.update_layout(
    xaxis_title="Fecha y Hora", 
    yaxis_title="Corriente (Amperios)", 
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis=dict(range=[0, max(230, df_filtrado['Corriente_Salida_A'].max() + 10)]),
    xaxis=dict(range=rango_tiempo) # <--- AÑADIDO ESTO
)

st.plotly_chart(fig2, width="stretch", key="grafica_corriente")

# ==========================================
# 9. GRÁFICA DE FRECUENCIA VS TIEMPO CON LÍMITES DE DISEÑO
# ==========================================
st.markdown("---")
st.subheader("Frecuencia v.s. Tiempo")

fig3 = go.Figure()

fig3.add_trace(go.Scattergl(x=df_filtrado.index, y=df_filtrado['Frecuencia_Hz'],
                         mode='lines', name='Frecuencia Real (Hz)',
                         line=dict(color='#ab63fa', width=1.5)))

# Línea de Frecuencia Nominal (60Hz)
fig3.add_hline(y=60, line_dash="dot", line_color="#50C878", line_width=2,
               annotation_text="FRECUENCIA NOMINAL (60 Hz)", 
               annotation_position="bottom right", annotation_font_color="#50C878")

# Línea de Operación Estable Mínima (50Hz)
fig3.add_hline(y=50, line_dash="dot", line_color="#ffa15a", line_width=1,
               annotation_text="OPERACIÓN ESTABLE TÍPICA (50 Hz)", 
               annotation_position="bottom right", annotation_font_color="#ffa15a")

fig3.update_layout(
    xaxis_title="Fecha y Hora", 
    yaxis_title="Frecuencia (Hz)", 
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    yaxis=dict(range=[0, 70]),
    xaxis=dict(range=rango_tiempo) 
)

st.plotly_chart(fig3, width="stretch", key="grafica_frecuencia")

# ==========================================
# 10. REGISTRO DE EVENTOS Y ALERTAS (LOG MULTIFILTRO)
# ==========================================
st.markdown("---")
st.subheader("📋 Registro de Alertas y Eventos")

# 10.1 Botones de Filtro (Agregamos Normal y Precaución)
filtro_log = st.radio(
    "Filtro:",
    ["Todas", "Normal", "Precaución", "Críticas", "IA"],
    horizontal=True,
    label_visibility="collapsed"
)

# 10.2 Generador Inteligente de Eventos
eventos = []

# Filtramos solo cuando la faja está "Operando" para no llenar el log de ceros
df_operando_log = df_filtrado[df_filtrado['Estado'] == 'Operando']

for tiempo, fila in df_operando_log.iterrows():
    corriente = fila['Corriente_Salida_A']
    
    # A. EVENTOS DE IA (Modelo Predictivo)
    if fila['Alarma_Modelo'] == -1:
        eventos.append({
            'tiempo': tiempo,
            'tipo': 'IA',
            'icono': '🧠',
            'color': '#A855F7', # Morado
            'titulo': 'Alerta Predictiva (IA)',
            'desc': f"Anomalía detectada. Ratio de esfuerzo elevado: {fila['Ratio_Esfuerzo']:.2f}",
            'sub': 'IA · MODELO PREDICTIVO'
        })
    
    # B. EVENTOS FÍSICOS (Estados de Corriente)
    if corriente >= 200:
        eventos.append({
            'tiempo': tiempo,
            'tipo': 'CRITICAS',
            'icono': '🔴',
            'color': '#EF4444', # Rojo
            'titulo': 'Sobreesfuerzo Crítico',
            'desc': f"Corriente de {corriente:.1f}A supera umbral crítico (>200A)",
            'sub': 'FÍSICO · CRÍTICO'
        })
    elif corriente >= 180:
        eventos.append({
            'tiempo': tiempo,
            'tipo': 'PRECAUCION',
            'icono': '🟡',
            'color': '#FBBF24', # Amarillo
            'titulo': 'Alta Carga Detectada',
            'desc': f"Corriente de {corriente:.1f}A en zona de precaución (>180A)",
            'sub': 'FÍSICO · PRECAUCIÓN'
        })
    else:
        eventos.append({
            'tiempo': tiempo,
            'tipo': 'NORMAL',
            'icono': '🟢',
            'color': '#10B981', # Verde
            'titulo': 'Operación Nominal',
            'desc': f"Corriente estable y segura a {corriente:.1f}A",
            'sub': 'FÍSICO · ESTABLE'
        })

# Ordenamos la lista para que el evento más reciente salga arriba
eventos = sorted(eventos, key=lambda x: x['tiempo'], reverse=True)

# 10.3 Aplicar el Filtro Seleccionado
if filtro_log == "Normal":
    eventos_mostrar = [e for e in eventos if e['tipo'] == 'NORMAL']
elif filtro_log == "Precaución":
    eventos_mostrar = [e for e in eventos if e['tipo'] == 'PRECAUCION']
elif filtro_log == "Críticas":
    eventos_mostrar = [e for e in eventos if e['tipo'] == 'CRITICAS']
elif filtro_log == "IA":
    eventos_mostrar = [e for e in eventos if e['tipo'] == 'IA']
else:
    eventos_mostrar = eventos # Opción "Todas"

# 👉 EL FRENO DE SEGURIDAD: Mostramos solo los 100 eventos más recientes
LIMITE_EVENTOS = 50
eventos_mostrar = eventos_mostrar[:LIMITE_EVENTOS]

# 10.4 Renderizado del Log con Scroll Automático
with st.container(height=400):
    if len(eventos_mostrar) == 0:
        st.success(f"✅ Sistema estable. No hay alertas para la categoría '{filtro_log}'.")
    else:
        # Avisamos al operador si hay más eventos ocultos
        if len(eventos) > LIMITE_EVENTOS and filtro_log == "Todas":
             st.caption(f"Mostrando los últimos {LIMITE_EVENTOS} eventos")
             
        for ev in eventos_mostrar:
            hora_str = ev['tiempo'].strftime("%H:%M") 
            fecha_str = ev['tiempo'].strftime("%Y-%m-%d")
            
           # Tarjeta HTML adaptada AUTOMÁTICAMENTE al tema (Claro/Oscuro/Sistema)
            tarjeta_html = f"""<div style="background-color: var(--secondary-background-color); border-left: 4px solid {ev['color']}; padding: 12px; margin-bottom: 8px; border-radius: 4px; display: flex; justify-content: space-between; font-family: monospace; border-top: 1px solid rgba(128,128,128,0.2); border-right: 1px solid rgba(128,128,128,0.2); border-bottom: 1px solid rgba(128,128,128,0.2);"><div style="display: flex; flex-direction: column;"><div style="display: flex; align-items: center; margin-bottom: 4px;"><span style="font-size: 16px;">{ev['icono']}</span><span style="color: var(--text-color); font-weight: bold; margin-left: 8px; font-size: 14px;">{ev['titulo']}</span></div><div style="color: var(--text-color); opacity: 0.8; font-size: 13px; margin-left: 28px; margin-bottom: 6px;">{ev['desc']}</div><div style="color: {ev['color']}; font-size: 11px; font-weight: bold; margin-left: 28px;">{ev['sub']}</div></div><div style="color: var(--text-color); opacity: 0.6; font-size: 12px; text-align: right;">{fecha_str}<br>{hora_str}</div></div>"""
            
            st.markdown(tarjeta_html, unsafe_allow_html=True)
