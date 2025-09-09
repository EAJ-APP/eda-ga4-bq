import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import TimeoutError
import warnings

# Suprimir warnings específicos de Plotly/Pandas
warnings.filterwarnings("ignore", category=FutureWarning, 
                      message="When grouping with a length-1 list-like.*")

# ===== 1. CONFIGURACIÓN INICIAL =====
def check_dependencies():
    """Verifica dependencias esenciales"""
    try:
        import db_dtypes, pandas, google.cloud.bigquery, plotly
    except ImportError as e:
        st.error(f"❌ Error: {str(e)}. Actualiza requirements.txt")
        st.stop()

# ===== 2. MANEJO DE ERRORES =====
def handle_bq_error(e, query=None):
    """Muestra errores de BigQuery de forma legible"""
    error_msg = f"""
    🚨 **Error en BigQuery**:
    ```python
    {str(e)}
    ```
    """
    if query:
        error_msg += f"\n**Consulta**:\n```sql\n{query}\n```"
    st.error(error_msg)
    st.stop()

# ===== 3. CONEXIÓN A BIGQUERY =====
def get_bq_client(credentials_path=None):
    """Crea cliente de BigQuery con manejo de errores simple"""
    try:
        if credentials_path:
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            creds_dict = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
        return bigquery.Client(credentials=credentials)
    except Exception as e:
        handle_bq_error(e)

# ===== 4. EJECUCIÓN DE CONSULTAS =====
def run_query(client, query, timeout=30):
    """Versión simplificada sin ServiceError"""
    try:
        query_job = client.query(query)
        return query_job.result(timeout=timeout).to_dataframe(create_bqstorage_client=False)
    except TimeoutError:
        handle_bq_error("⏳ Timeout: La consulta tardó demasiado. Filtra más datos.", query)
    except Exception as e:
        handle_bq_error(e, query)

# ===== 5. CONSULTAS ESPECÍFICAS =====
def generar_query_consentimiento_basico(project, dataset, start_date, end_date):
    """Consulta básica de consentimiento"""
    return f"""
    SELECT
      privacy_info.analytics_storage AS analytics_storage_status,
      privacy_info.ads_storage AS ads_storage_status,
      COUNT(*) AS total_events,
      COUNT(DISTINCT user_pseudo_id) AS total_users,
      COUNT(DISTINCT CONCAT(user_pseudo_id, '-', 
        (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id'))) AS total_sessions
    FROM `{project}.{dataset}.events_*`
    WHERE _TABLE_SUFFIX BETWEEN '{start_date.strftime('%Y%m%d')}' AND '{end_date.strftime('%Y%m%d')}'
    GROUP BY 1, 2
    ORDER BY 3 DESC
    """

def generar_query_consentimiento_por_dispositivo(project, dataset, start_date, end_date):
    """Consulta optimizada que garantiza datos diferentes"""
    return f"""
    WITH base_data AS (
      SELECT
        device.category AS device_type,
        CASE
          WHEN privacy_info.analytics_storage IS NULL THEN 'null'
          WHEN LOWER(CAST(privacy_info.analytics_storage AS STRING)) IN ('true', 'yes', '1') THEN 'true'
          ELSE 'false'
        END AS analytics_status,
        CASE
          WHEN privacy_info.ads_storage IS NULL THEN 'null'
          WHEN LOWER(CAST(privacy_info.ads_storage AS STRING)) IN ('true', 'yes', '1') THEN 'true'
          ELSE 'false'
        END AS ads_status,
        user_pseudo_id,
        (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id') AS session_id
      FROM `{project}.{dataset}.events_*`
      WHERE _TABLE_SUFFIX BETWEEN '{start_date.strftime('%Y%m%d')}' AND '{end_date.strftime('%Y%m%d')}'
    )
    SELECT
      device_type,
      analytics_status,
      ads_status,
      COUNT(*) AS total_events,
      COUNT(DISTINCT user_pseudo_id) AS total_users,
      COUNT(DISTINCT CONCAT(user_pseudo_id, '-', session_id)) AS total_sessions
    FROM base_data
    GROUP BY 1, 2, 3
    ORDER BY device_type, total_events DESC
    """

def generar_query_consentimiento_real(project, dataset, start_date, end_date):
    """Nueva consulta para porcentaje real de consentimiento"""
    return f"""
    SELECT
        CASE
            WHEN privacy_info.analytics_storage IS NULL THEN 'No Definido'
            WHEN LOWER(CAST(privacy_info.analytics_storage AS STRING)) IN ('false', 'no', '0') THEN 'Denegado'
            ELSE 'Aceptado'
        END AS consent_status,
        COUNT(*) AS total_events,
        ROUND(COUNT(*) / SUM(COUNT(*)) OVER() * 100, 2) AS event_percentage
    FROM `{project}.{dataset}.events_*`
    WHERE _TABLE_SUFFIX BETWEEN '{start_date.strftime('%Y%m%d')}' AND '{end_date.strftime('%Y%m%d')}'
    GROUP BY 1
    ORDER BY total_events DESC
    """

def generar_query_comparativa_eventos(project, dataset, start_date, end_date):
    """Consulta para comparativa completa de eventos de ecommerce"""
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    return f"""
    SELECT
      event_date,
      event_name,
      COUNT(*) AS total_events,
      COUNT(DISTINCT user_pseudo_id) AS unique_users
    FROM `{project}.{dataset}.events_*`
    WHERE event_name IN ('page_view', 'view_item', 'add_to_cart', 'begin_checkout', 'purchase')
      AND _TABLE_SUFFIX BETWEEN '{start_date_str}' AND '{end_date_str}'
    GROUP BY event_date, event_name
    ORDER BY event_date, total_events DESC
    """

# ===== 6. VISUALIZACIONES =====
def mostrar_consentimiento_basico(df):
    """Visualización para consulta básica de consentimiento con porcentajes"""
    st.subheader("📋 Datos Crudos")
    
    # Calcular totales para porcentajes
    total_eventos = df['total_events'].sum()
    total_usuarios = df['total_users'].sum()
    total_sesiones = df['total_sessions'].sum()
    
    # Crear copia del DataFrame para no modificar el original
    df_mostrar = df.copy()
    
    # Calcular porcentajes
    df_mostrar['% eventos'] = (df_mostrar['total_events'] / total_eventos * 100).round(2).astype(str) + '%'
    df_mostrar['% usuarios'] = (df_mostrar['total_users'] / total_usuarios * 100).round(2).astype(str) + '%'
    df_mostrar['% sesiones'] = (df_mostrar['total_sessions'] / total_sesiones * 100).round(2).astype(str) + '%'
    
    # Reordenar columnas
    columnas = ['analytics_storage_status', 'ads_storage_status', 
                'total_events', '% eventos',
                'total_users', '% usuarios',
                'total_sessions', '% sesiones']
    
    st.dataframe(df_mostrar[columnas].style.format({
        'total_events': '{:,}',
        'total_users': '{:,}',
        'total_sessions': '{:,}'
    }))
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(df, names='analytics_storage_status', 
                     values='total_events', title='Eventos por Consentimiento Analytics')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.bar(df, x='ads_storage_status', y='total_users',
                     title='Usuarios Únicos por Consentimiento Ads')
        st.plotly_chart(fig2, use_container_width=True)

def mostrar_consentimiento_por_dispositivo(df):
    """Visualización corregida que muestra datos diferentes en cada pestaña"""
    st.subheader("📱 Consentimiento por Dispositivo (Detallado)")
    
    if df.empty:
        st.warning("No hay datos disponibles para el rango seleccionado")
        return
    
    # Preprocesamiento
    df['device_type'] = df['device_type'].str.capitalize()
    consent_map = {'true': 'Consentido', 'false': 'No Consentido', 'null': 'No Definido'}
    
    # Orden de dispositivos por eventos totales
    device_order = df.groupby('device_type')['total_events'].sum().sort_values(ascending=False).index
    
    tab1, tab2 = st.tabs(["Analytics Storage", "Ads Storage"])
    
    with tab1:
        # Filtramos y preparamos datos específicos para Analytics
        df_analytics = df[['device_type', 'analytics_status', 'total_events']].copy()
        df_analytics['consent_status'] = df_analytics['analytics_status'].map(consent_map)
        
        # Agrupamos para el gráfico
        df_analytics_grouped = df_analytics.groupby(['device_type', 'consent_status'])['total_events'].sum().reset_index()
        
        fig_analytics = px.bar(
            df_analytics_grouped,
            x='device_type',
            y='total_events',
            color='consent_status',
            category_orders={"device_type": list(device_order)},
            barmode='stack',
            title='Consentimiento Analytics por Dispositivo',
            labels={'device_type': 'Dispositivo', 'total_events': 'Eventos'},
            color_discrete_map={
                'Consentido': '#4CAF50',
                'No Consentido': '#F44336',
                'No Definido': '#9E9E9E'
            }
        )
        st.plotly_chart(fig_analytics, use_container_width=True)
        
        st.write("Datos Analytics:")
        st.dataframe(df_analytics_grouped.pivot(index='device_type', columns='consent_status', values='total_events'))
    
    with tab2:
        # Filtramos y preparamos datos específicos para Ads
        df_ads = df[['device_type', 'ads_status', 'total_events']].copy()
        df_ads['consent_status'] = df_ads['ads_status'].map(consent_map)
        
        # Agrupamos para el gráfico
        df_ads_grouped = df_ads.groupby(['device_type', 'consent_status'])['total_events'].sum().reset_index()
        
        fig_ads = px.bar(
            df_ads_grouped,
            x='device_type',
            y='total_events',
            color='consent_status',
            category_orders={"device_type": list(device_order)},
            barmode='stack',
            title='Consentimiento Ads por Dispositivo',
            labels={'device_type': 'Dispositivo', 'total_events': 'Eventos'},
            color_discrete_map={
                'Consentido': '#4CAF50',
                'No Consentido': '#F44336',
                'No Definido': '#9E9E9E'
            }
        )
        st.plotly_chart(fig_ads, use_container_width=True)
        
        st.write("Datos Ads:")
        st.dataframe(df_ads_grouped.pivot(index='device_type', columns='consent_status', values='total_events'))
    
    # Estadísticas comparativas
    st.subheader("📊 Comparativa de Consentimientos")
    col1, col2 = st.columns(2)
    
    with col1:
        analytics_true = df[df['analytics_status'] == 'true']['total_events'].sum()
        st.metric("Eventos con Consentimiento Analytics", f"{analytics_true:,}")
    
    with col2:
        ads_true = df[df['ads_status'] == 'true']['total_events'].sum()
        st.metric("Eventos con Consentimiento Ads", f"{ads_true:,}")

def mostrar_consentimiento_real(df):
    """Nueva visualización para porcentaje real de consentimiento"""
    st.subheader("🔍 Porcentaje Real de Consentimiento (Todos los Eventos)")
    
    # Mapeo de estados a colores
    status_colors = {
        'Aceptado': '#4CAF50',
        'Denegado': '#F44336',
        'No Definido': '#FFC107'
    }
    
    # Gráfico de torta
    fig = px.pie(df, 
                 names='consent_status', 
                 values='total_events',
                 color='consent_status',
                 color_discrete_map=status_colors,
                 title='Distribución Real del Consentimiento')
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar tabla con datos crudos
    st.dataframe(df.style.format({
        'total_events': '{:,}',
        'event_percentage': '{:.2f}%'
    }))
    
    # Calcular y mostrar el % de eventos SIN consentimiento (Denegado + No Definido)
    denied_pct = df[df['consent_status'].isin(['Denegado', 'No Definido'])]['event_percentage'].sum()
    st.metric("📉 Eventos sin consentimiento (Real)", f"{denied_pct:.2f}%")

def mostrar_comparativa_eventos(df):
    """Visualización para comparativa completa de eventos (con funnel como antes)"""
    st.subheader("📊 Funnel de Ecommerce")
    
    if df.empty:
        st.warning("No hay datos disponibles para el rango seleccionado")
        return
    
    # Agregar datos por tipo de evento (suma total)
    event_totals = df.groupby('event_name').agg({
        'total_events': 'sum',
        'unique_users': 'sum'
    }).reset_index()
    
    # Crear datos para el funnel
    funnel_data = {
        'event_name': ['page_view', 'view_item', 'add_to_cart', 'begin_checkout', 'purchase'],
        'total_events': [
            event_totals[event_totals['event_name'] == 'page_view']['total_events'].sum(),
            event_totals[event_totals['event_name'] == 'view_item']['total_events'].sum(),
            event_totals[event_totals['event_name'] == 'add_to_cart']['total_events'].sum(),
            event_totals[event_totals['event_name'] == 'begin_checkout']['total_events'].sum(),
            event_totals[event_totals['event_name'] == 'purchase']['total_events'].sum()
        ]
    }
    
    funnel_df = pd.DataFrame(funnel_data)
    
    # Mostrar tabla con datos crudos
    st.dataframe(df.style.format({
        'total_events': '{:,}',
        'unique_users': '{:,}'
    }))
    
    # Calcular tasas de conversión
    page_views = funnel_df[funnel_df['event_name'] == 'page_view']['total_events'].values[0]
    view_items = funnel_df[funnel_df['event_name'] == 'view_item']['total_events'].values[0]
    add_to_cart = funnel_df[funnel_df['event_name'] == 'add_to_cart']['total_events'].values[0]
    begin_checkout = funnel_df[funnel_df['event_name'] == 'begin_checkout']['total_events'].values[0]
    purchases = funnel_df[funnel_df['event_name'] == 'purchase']['total_events'].values[0]
    
    # Mostrar métricas de conversión
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        view_item_rate = (view_items / page_views * 100) if page_views > 0 else 0
        st.metric("Tasa View Item", f"{view_item_rate:.2f}%")
    with col2:
        add_to_cart_rate = (add_to_cart / view_items * 100) if view_items > 0 else 0
        st.metric("Tasa Add to Cart", f"{add_to_cart_rate:.2f}%")
    with col3:
        checkout_rate = (begin_checkout / view_items * 100) if view_items > 0 else 0
        st.metric("Tasa Checkout", f"{checkout_rate:.2f}%")
    with col4:
        purchase_rate = (purchases / view_items * 100) if view_items > 0 else 0
        st.metric("Tasa Compra", f"{purchase_rate:.2f}%")
    
    # Gráfico de funnel
    fig_funnel = go.Figure(go.Funnel(
        y=["Page Views", "View Item", "Add to Cart", "Begin Checkout", "Purchase"],
        x=[page_views, view_items, add_to_cart, begin_checkout, purchases],
        textinfo="value+percent initial",
        opacity=0.8,
        marker={"color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]}
    ))
    
    fig_funnel.update_layout(title="Funnel de Conversión de Ecommerce")
    st.plotly_chart(fig_funnel, use_container_width=True)
    
    # Gráfico de tendencia de conversiones
    df_pivot = df.pivot_table(
        index='event_date', 
        columns='event_name', 
        values='total_events', 
        aggfunc='sum'
    ).fillna(0).reset_index()
    
    fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Añadir conversiones (barras)
    fig_trend.add_trace(
        go.Bar(x=df_pivot['event_date'], y=df_pivot['purchase'], name="Compras", opacity=0.7),
        secondary_y=False,
    )
    
    # Calcular tasa de compra diaria
    df_pivot['purchase_rate_daily'] = (df_pivot['purchase'] / df_pivot['view_item'] * 100).fillna(0)
    
    # Añadir tasas de conversión (líneas)
    fig_trend.add_trace(
        go.Scatter(x=df_pivot['event_date'], y=df_pivot['purchase_rate_daily'], 
                  name="Tasa de Compra", line=dict(color='firebrick', width=3)),
        secondary_y=True,
    )
    
    # Configurar ejes
    fig_trend.update_xaxes(title_text="Fecha")
    fig_trend.update_yaxes(title_text="Compras", secondary_y=False)
    fig_trend.update_yaxes(title_text="Tasa de Compra (%)", secondary_y=True)
    fig_trend.update_layout(title="Tendencia de Compras y Tasa de Conversión")
    
    st.plotly_chart(fig_trend, use_container_width=True)

# ===== 7. INTERFAZ PRINCIPAL =====
def show_cookies_tab(client, project, dataset, start_date, end_date):
    """Pestaña de Cookies con consultas separadas"""
    with st.expander("🛡️ Consentimiento Básico", expanded=True):
        if st.button("Ejecutar Análisis Básico", key="btn_consent_basic"):
            with st.spinner("Calculando consentimientos..."):
                query = generar_query_consentimiento_basico(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_consentimiento_basico(df)
    
    with st.expander("📱 Consentimiento por Dispositivo", expanded=True):
        if st.button("Ejecutar Análisis por Dispositivo", key="btn_consent_device"):
            with st.spinner("Analizando dispositivos..."):
                query = generar_query_consentimiento_por_dispositivo(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_consentimiento_por_dispositivo(df)
    
    with st.expander("🔍 Porcentaje Real de Consentimiento", expanded=True):
        if st.button("Calcular Consentimiento Real", key="btn_consent_real"):
            with st.spinner("Analizando todos los eventos..."):
                query = generar_query_consentimiento_real(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_consentimiento_real(df)

def show_ecommerce_tab(client, project, dataset, start_date, end_date):
    """Pestaña de Ecommerce con análisis de eventos"""
    with st.expander("📊 Funnel de Conversión", expanded=True):
        if st.button("Ejecutar Análisis de Ecommerce", key="btn_ecommerce"):
            with st.spinner("Analizando funnel de conversión..."):
                query = generar_query_comparativa_eventos(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_comparativa_eventos(df)

def main():
    check_dependencies()
    st.set_page_config(page_title="GA4 Explorer", layout="wide")
    st.title("📊 Análisis Exploratorio GA4")

    # --- Sidebar ---
    with st.sidebar:
        st.header("🔧 Configuración")
        development_mode = st.toggle("Modo desarrollo (usar JSON local)")
        
        if development_mode:
            creds_file = st.file_uploader("Sube credenciales JSON", type=["json"])
            if creds_file:
                with open("temp_creds.json", "wb") as f:
                    f.write(creds_file.getvalue())
                st.session_state.creds = "temp_creds.json"
        elif "gcp_service_account" not in st.secrets:
            st.error("⚠️ Configura los Secrets en Streamlit Cloud")
            st.stop()

        st.header("📅 Rango de Fechas")
        start_date = st.date_input("Fecha inicio", value=pd.to_datetime("2023-01-01"), key="global_start_date")
        end_date = st.date_input("Fecha fin", value=pd.to_datetime("today"), key="global_end_date")

    # --- Conexión ---
    client = get_bq_client(
        st.session_state.creds if development_mode and "creds" in st.session_state else None
    )

    # --- Selectores ---
    try:
        projects = [p.project_id for p in client.list_projects()]
        selected_project = st.sidebar.selectbox("Proyecto", projects)
        datasets = [d.dataset_id for d in client.list_datasets(selected_project)]
        selected_dataset = st.sidebar.selectbox("Dataset GA4", datasets)
    except Exception as e:
        handle_bq_error(e)

    # --- Tabs Principales ---
    tab_titles = [
        "🍪 Cookies y Privacidad",
        "🛒 Ecommerce", 
        "📈 Adquisición",
        "🎯 Eventos",
        "👥 Usuarios",
        "🕒 Sesiones"
    ]
    tab_ids = ["cookies", "ecommerce", "acquisition", "events", "users", "sessions"]
    
    tabs = st.tabs(tab_titles)
    
    for tab, tab_id in zip(tabs, tab_ids):
        with tab:
            st.header(f"Análisis de {tab_id.capitalize()}")
            if tab_id == "cookies":
                show_cookies_tab(client, selected_project, selected_dataset, start_date, end_date)
            elif tab_id == "ecommerce":
                show_ecommerce_tab(client, selected_project, selected_dataset, start_date, end_date)
            else:
                st.info(f"🔧 Sección en desarrollo. Próximamente: consultas para {tab_id}")

if __name__ == "__main__":
    main()
