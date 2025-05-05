import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import time
from concurrent.futures import TimeoutError

# ===== 1. CONFIGURACIÓN INICIAL =====
def check_dependencies():
    """Verifica dependencias esenciales"""
    try:
        import db_dtypes, pandas, google.cloud.bigquery
    except ImportError as e:
        st.error(f"❌ Error: {str(e)}. Actualiza requirements.txt")
        st.stop()

# ===== 2. MANEJO DE ERRORES SIMPLIFICADO =====
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
        return query_job.result(timeout=timeout).to_dataframe()
    except TimeoutError:
        handle_bq_error("⏳ Timeout: La consulta tardó demasiado. Filtra más datos.", query)
    except Exception as e:
        handle_bq_error(e, query)

# ===== 5. INTERFAZ PRINCIPAL CON TABS =====
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🍪 Cookies y Privacidad",
        "🛒 Ecommerce", 
        "📈 Adquisición",
        "🎯 Eventos",
        "👥 Usuarios",
        "🕒 Sesiones"
    ])

    # Contenido idéntico temporal en todas las tabs
    with tab1:
        st.header("Consulta básica")
        show_common_interface(client, selected_project, selected_dataset)
    
    with tab2:
        st.header("Consulta básica")
        show_common_interface(client, selected_project, selected_dataset)
    
    with tab3:
        st.header("Consulta básica")
        show_common_interface(client, selected_project, selected_dataset)
    
    with tab4:
        st.header("Consulta básica")
        show_common_interface(client, selected_project, selected_dataset)
    
    with tab5:
        st.header("Consulta básica")
        show_common_interface(client, selected_project, selected_dataset)
    
    with tab6:
        st.header("Consulta básica")
        show_common_interface(client, selected_project, selected_dataset)

# ... (imports y funciones anteriores se mantienen igual)

def show_common_interface(client, project, dataset, tab_id):
    """Muestra la interfaz común en todas las tabs (temporal)"""
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio", 
                                 value=pd.to_datetime("2023-01-01"), 
                                 key=f"start_date_{tab_id}")
    with col2:
        end_date = st.date_input("Fecha fin", 
                               value=pd.to_datetime("today"), 
                               key=f"end_date_{tab_id}")
    
    if st.button("Ejecutar consulta", key=f"btn_{tab_id}"):
        query = f"""
            SELECT 
                event_name,
                COUNT(*) as event_count
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
    # ... (configuración inicial y sidebar se mantiene igual)

    # --- Tabs ---
    tabs = st.tabs([
        "🍪 Cookies y Privacidad",
        "🛒 Ecommerce", 
        "📈 Adquisición",
        "🎯 Eventos",
        "👥 Usuarios",
        "🕒 Sesiones"
    ])

    # Asignamos un ID único a cada tab
    tab_ids = ["cookies", "ecommerce", "acquisition", "events", "users", "sessions"]

    # Contenido para cada tab
    for tab, tab_id in zip(tabs, tab_ids):
        with tab:
            st.header(f"Análisis de {tab_id.capitalize()}")
            show_common_interface(client, selected_project, selected_dataset, tab_id)

if __name__ == "__main__":
    main()
