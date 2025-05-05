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
def generar_query_cookies(project, dataset, start_date, end_date):
    """Consulta de consentimiento de cookies"""
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

def mostrar_resultados_cookies(df):
    """Visualización para datos de cookies"""
    # Preprocesamiento
    df['analytics_status'] = df['analytics_storage_status'].map(
        {True: 'Aceptado', False: 'Rechazado', None: 'No especificado'})
    df['ads_status'] = df['ads_storage_status'].map(
        {True: 'Aceptado', False: 'Rechazado', None: 'No especificado'})
    
    # Tabla
    st.subheader("📋 Datos Crudos")
    st.dataframe(df)
    
    # Gráficos
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🍪 Consentimiento Analytics")
        fig1 = px.pie(df, names='analytics_status', values='total_users')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("📢 Consentimiento Ads")
        fig2 = px.pie(df, names='ads_status', values='total_sessions')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Gráfico combinado
    st.subheader("📈 Comparativa Completa")
    fig3 = px.bar(df, 
                 x='analytics_status', 
                 y='total_events', 
                 color='ads_status',
                 barmode='group',
                 labels={'total_events': 'Eventos totales'})
    st.plotly_chart(fig3, use_container_width=True)

# ===== 6. INTERFAZ PRINCIPAL =====
def show_tab_interface(client, project, dataset, tab_id, start_date, end_date):
    """Interfaz específica para cada tab"""
    if tab_id == "cookies":
        query = generar_query_cookies(project, dataset, start_date, end_date)
        df = run_query(client, query)
        mostrar_resultados_cookies(df)
    else:
        # Consulta genérica para otras pestañas
        query = f"""
            SELECT event_name, COUNT(*) as event_count
            FROM `{project}.{dataset}.events_*`
            WHERE _TABLE_SUFFIX BETWEEN '{start_date.strftime('%Y%m%d')}' 
                AND '{end_date.strftime('%Y%m%d')}'
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 20
        """
        df = run_query(client, query)
        st.dataframe(df)
        st.bar_chart(df.set_index("event_name"))

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

    # --- Tabs ---
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
            show_tab_interface(client, selected_project, selected_dataset, tab_id, start_date, end_date)

if __name__ == "__main__":
    main()
