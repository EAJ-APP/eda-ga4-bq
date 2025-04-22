import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import os

# --- Verificación de dependencias ---
def check_dependencies():
    """Verifica que los paquetes necesarios estén instalados"""
    try:
        import db_dtypes  # noqa
    except ImportError:
        st.error("""Falta el paquete db-dtypes. Por favor actualiza tus dependencias:
                 \n1. Agrega 'db-dtypes==1.2.0' a requirements.txt
                 \n2. Reinicia la app en Streamlit Cloud""")
        st.stop()

# --- Configuración BigQuery ---
def get_bq_client(credentials_path=None):
    """Obtiene el cliente de BigQuery"""
    try:
        if credentials_path:
            # Para desarrollo local con archivo JSON
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        else:
            # Para producción en Streamlit Cloud
            creds_dict = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
        
        return bigquery.Client(credentials=credentials)
    except Exception as e:
        st.error(f"Error al conectar con BigQuery: {str(e)}")
        st.stop()

def run_query(client, query):
    """Ejecuta una consulta y devuelve un DataFrame."""
    try:
        query_job = client.query(query)
        return query_job.to_dataframe(
            create_bqstorage_client=False,
            dtypes={"event_count": "int64"}  # Tipado explícito para columnas conocidas
        )
    except Exception as e:
        st.error(f"Error en la consulta: {str(e)}")
        st.stop()

# --- Interfaz Streamlit ---
def main():
    check_dependencies()  # Verificar paquetes primero
    
    st.set_page_config(page_title="GA4 Explorer", layout="wide")
    st.title("📊 Análisis Exploratorio GA4")

    # --- Sidebar: Configuración ---
    with st.sidebar:
        st.header("🔑 Configuración")
        
        # Modo desarrollo/producción
        development_mode = st.toggle("Modo desarrollo (usar archivo JSON)")
        
        if development_mode:
            creds_file = st.file_uploader("Sube tus credenciales JSON", type=["json"])
            if creds_file:
                with open("temp_creds.json", "wb") as f:
                    f.write(creds_file.getvalue())
                st.session_state.creds = "temp_creds.json"
                st.success("Credenciales cargadas!")
        else:
            if "gcp_service_account" not in st.secrets:
                st.error("Configure los secrets en Streamlit Cloud")
                st.stop()

    # --- Conexión a BigQuery ---
    if development_mode and "creds" in st.session_state:
        client = get_bq_client(st.session_state.creds)
    else:
        client = get_bq_client()

    # Obtener proyectos y datasets
    try:
        projects = [p.project_id for p in client.list_projects()]
        selected_project = st.sidebar.selectbox("Proyecto BigQuery", projects, key="project_select")
        
        datasets = [d.dataset_id for d in client.list_datasets(selected_project)]
        selected_dataset = st.sidebar.selectbox("Dataset GA4", datasets, key="dataset_select")
        
        st.session_state.client = client
        st.session_state.project = selected_project
        st.session_state.dataset = selected_dataset
    except Exception as e:
        st.error(f"No se pudo obtener la lista de proyectos/datasets: {str(e)}")
        st.stop()

    # --- Panel principal ---
    st.header("🔍 Consulta Básica")
    
    # Selector de fechas
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio", value=pd.to_datetime("2023-01-01"), key="start_date")
    with col2:
        end_date = st.date_input("Fecha fin", value=pd.to_datetime("today"), key="end_date")
    
    if st.button("Obtener eventos principales"):
        query = f"""
            SELECT 
                event_name,
                COUNT(*) as event_count
            FROM `{st.session_state.project}.{st.session_state.dataset}.events_*`
            WHERE _TABLE_SUFFIX BETWEEN '{start_date.strftime('%Y%m%d')}' 
                AND '{end_date.strftime('%Y%m%d')}'
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 20
        """
        df = run_query(st.session_state.client, query)
        
        # Mostrar resultados
        st.dataframe(df)
        st.bar_chart(df.set_index("event_name"))
        
        # Opcional: Mostrar metadatos
        with st.expander("📊 Metadatos de los resultados"):
            st.write("Tipos de datos:", df.dtypes)
            st.write("Total de filas:", len(df))

if __name__ == "__main__":
    main()
