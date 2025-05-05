import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import plotly.express as px
from concurrent.futures import TimeoutError

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
    """Consulta corregida con CAST explícito para booleanos"""
    return f"""
    SELECT
      device.category AS device_type,
      CASE 
        WHEN privacy_info.analytics_storage IS NULL THEN 'null'
        WHEN CAST(privacy_info.analytics_storage AS STRING) = 'true' THEN 'true'
        ELSE 'false'
      END AS analytics_storage_status,
      CASE 
        WHEN privacy_info.ads_storage IS NULL THEN 'null'
        WHEN CAST(privacy_info.ads_storage AS STRING) = 'true' THEN 'true'
        ELSE 'false'
      END AS ads_storage_status,
      COUNT(*) AS total_events,
      COUNT(DISTINCT user_pseudo_id) AS total_users,
      COUNT(DISTINCT CONCAT(user_pseudo_id, '-', 
        (SELECT value.int_value FROM UNNEST(event_params) WHERE key = 'ga_session_id'))) AS total_sessions
    FROM `{project}.{dataset}.events_*`
    WHERE _TABLE_SUFFIX BETWEEN '{start_date.strftime('%Y%m%d')}' AND '{end_date.strftime('%Y%m%d')}'
    GROUP BY 1, 2, 3
    ORDER BY device_type, total_events DESC
    """

def generar_query_estimacion_usuarios(project, dataset, start_date, end_date):
    """Versión corregida que maneja correctamente los tipos de datos"""
    return f"""
    WITH ConsentFactors AS (
        SELECT 2.0 AS factor_value, 'true' AS consent_state
        UNION ALL
        SELECT 3.0 AS factor_value, 'false' AS consent_state
    ),
    EventData AS (
        SELECT
            CASE 
                WHEN privacy_info.analytics_storage IS NULL THEN 'false'
                ELSE LOWER(CAST(privacy_info.analytics_storage AS STRING))
            END AS consent_state,
            COUNT(1) AS total_events,
            COUNT(DISTINCT user_pseudo_id) AS distinct_users
        FROM `{project}.{dataset}.events_*`
        WHERE event_name = 'page_view'
          AND _TABLE_SUFFIX BETWEEN '{start_date.strftime('%Y%m%d')}' AND '{end_date.strftime('%Y%m%d')}'
        GROUP BY consent_state
    ),
    ConsentAnalysis AS (
        SELECT
            CASE 
                WHEN ed.consent_state = 'true' THEN 'Granted'
                ELSE 'Denied'
            END AS consent_state,
            ed.total_events,
            ed.distinct_users,
            CAST(ROUND(ed.total_events / cf.factor_value) AS INT64) AS estimated_users
        FROM EventData ed
        LEFT JOIN ConsentFactors cf ON 
            CASE 
                WHEN ed.consent_state = 'true' THEN 'true'
                ELSE 'false'
            END = cf.consent_state
    )
    SELECT
        consent_state,
        total_events,
        distinct_users,
        estimated_users,
        ROUND(SAFE_DIVIDE(total_events, SUM(total_events) OVER()), 4) AS event_share
    FROM ConsentAnalysis
    ORDER BY total_events DESC
    """

# ===== 6. VISUALIZACIONES =====
def mostrar_consentimiento_basico(df):
    """Visualización para consulta básica de consentimiento"""
    st.subheader("📋 Datos Crudos")
    st.dataframe(df)
    
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
    """Visualización completa de consentimiento por dispositivo"""
    st.subheader("📱 Consentimiento por Dispositivo (Detallado)")
    
    # Preprocesamiento
    df['device_type'] = df['device_type'].str.capitalize()
    df['analytics_storage_status'] = df['analytics_storage_status'].map({
        'true': 'Consentido', 
        'false': 'No Consentido',
        'null': 'No Definido'
    })
    df['ads_storage_status'] = df['ads_storage_status'].map({
        'true': 'Consentido', 
        'false': 'No Consentido',
        'null': 'No Definido'
    })
    
    # Gráficos en tabs
    tab1, tab2 = st.tabs(["Analytics Storage", "Ads Storage"])
    
    with tab1:
        fig_analytics = px.bar(df,
                             x='device_type',
                             y='total_events',
                             color='analytics_storage_status',
                             barmode='stack',
                             title='Consentimiento Analytics por Dispositivo',
                             labels={'device_type': 'Dispositivo', 'total_events': 'Eventos'},
                             color_discrete_map={
                                 'Consentido': '#4CAF50',
                                 'No Consentido': '#F44336',
                                 'No Definido': '#9E9E9E'
                             })
        st.plotly_chart(fig_analytics, use_container_width=True)
        
    with tab2:
        fig_ads = px.bar(df,
                       x='device_type',
                       y='total_events',
                       color='ads_storage_status',
                       barmode='stack',
                       title='Consentimiento Ads por Dispositivo',
                       labels={'device_type': 'Dispositivo', 'total_events': 'Eventos'},
                       color_discrete_map={
                           'Consentido': '#4CAF50',
                           'No Consentido': '#F44336',
                           'No Definido': '#9E9E9E'
                       })
        st.plotly_chart(fig_ads, use_container_width=True)
    
    # Tabla resumen completa
    st.subheader("📊 Datos Completos")
    st.dataframe(
        df.sort_values(['device_type', 'total_events'], ascending=[True, False])
    )

def mostrar_estimacion_usuarios(df):
    """Visualización para estimación de usuarios"""
    st.subheader("📊 Estimación de Usuarios Reales")
    df['consent_state'] = df['consent_state'].map({
        'Granted': 'Consentimiento Otorgado',
        'Denied': 'Consentimiento Denegado'
    })
    
    fig = px.bar(df, x='consent_state', y=['distinct_users', 'estimated_users'],
                barmode='group', title='Comparativa: Usuarios Detectados vs Estimados',
                labels={'value': 'Número de Usuarios', 'variable': 'Tipo'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric("📈 Porcentaje de Eventos sin Consentimiento", 
             f"{df[df['consent_state'] == 'Consentimiento Denegado']['event_share'].values[0]*100:.2f}%")

# ===== 7. INTERFAZ PRINCIPAL =====
def show_cookies_tab(client, project, dataset, start_date, end_date):
    """Pestaña de Cookies con múltiples consultas en acordeones"""
    with st.expander("🛡️ Consentimiento Básico", expanded=True):
        if st.button("Ejecutar Análisis Básico", key="btn_consent_basic"):
            with st.spinner("Calculando consentimientos..."):
                query = generar_query_consentimiento_basico(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_consentimiento_basico(df)
    
    with st.expander("📱 Consentimiento por Dispositivo"):
        if st.button("Ejecutar Análisis por Dispositivo", key="btn_consent_device"):
            with st.spinner("Analizando dispositivos..."):
                query = generar_query_consentimiento_por_dispositivo(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_consentimiento_por_dispositivo(df)
    
    with st.expander("📈 Estimación de Usuarios Reales"):
        if st.button("Ejecutar Estimación", key="btn_estimation"):
            with st.spinner("Estimando usuarios reales..."):
                query = generar_query_estimacion_usuarios(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_estimacion_usuarios(df)

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
            else:
                st.info(f"🔧 Sección en desarrollo. Próximamente: consultas para {tab_id}")

if __name__ == "__main__":
    main()
