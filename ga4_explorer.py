import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account

try:
    from google.api_core.exceptions import ServiceError
except ImportError:
    from google.cloud.exceptions import ServiceError  # Fallback alternativo
    
import pandas as pd
import time
from concurrent.futures import TimeoutError

# ===== 1. VERIFICACIÓN DE DEPENDENCIAS =====
def check_dependencies():
    """Verifica que los paquetes esenciales estén instalados"""
    try:
        import db_dtypes  # Necesario para BigQuery -> pandas
        import pandas as pd
        from google.cloud import bigquery
    except ImportError as e:
        st.error(f"❌ Error crítico: Falta paquete ({str(e)}). Actualiza requirements.txt y reinicia la app.")
        st.stop()

# ===== 2. VALIDACIÓN DE ESQUEMA =====
def validate_schema(df):
    """Valida que el DataFrame tenga las columnas esperadas"""
    required_columns = {
        'event_name': 'object',  # pandas dtype equivalente a string
        'event_count': 'int64'
    }
    
    for col, dtype in required_columns.items():
        if col not in df.columns:
            raise ValueError(f"Columna faltante: {col}")
        if str(df[col].dtype) != dtype:
            raise ValueError(f"Tipo incorrecto en {col}. Esperado: {dtype}, Obtenido: {df[col].dtype}")

# ===== 3. CONEXIÓN A BIGQUERY =====
def get_bq_client(credentials_path=None):
    """Crea cliente de BigQuery (local o en la nube)"""
    try:
        if credentials_path:  # Modo desarrollo (archivo JSON)
            credentials = service_account.Credentials.from_service_account_file(credentials_path)
        else:  # Modo producción (Streamlit Cloud Secrets)
            creds_dict = dict(st.secrets["gcp_service_account"])
            credentials = service_account.Credentials.from_service_account_info(creds_dict)
        return bigquery.Client(credentials=credentials)
    except Exception as e:
        st.error(f"🔌 Error de conexión: {str(e)}")
        st.stop()

# ===== 4. EJECUCIÓN ROBUSTA DE CONSULTAS =====
def run_query(client, query, timeout=30, max_retries=3):
    """Ejecuta SQL con timeout y reintentos para errores temporales"""
    for attempt in range(max_retries):
        try:
            query_job = client.query(query)
            df = query_job.result(timeout=timeout).to_dataframe(
                create_bqstorage_client=False  # Evita dependencia adicional
            )
            validate_schema(df)
            return df
        except TimeoutError:
            st.warning(f"⏳ Timeout (intento {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                st.error("⌛ Consulta demasiado larga. Filtra más datos o simplifica la query.")
                st.stop()
            time.sleep(2)  # Espera breve antes de reintentar
        except ServiceError as e:  # Errores temporales de Google Cloud
            st.warning(f"⚠️ Reintentando... (error temporal: {str(e)})")
            time.sleep(2 ** attempt)  # Espera exponencial
        except Exception as e:
            st.error(f"❌ Error irrecuperable:\n{str(e)}\n\nConsulta:\n```sql\n{query}\n```")
            st.stop()

# ===== 5. INTERFAZ PRINCIPAL =====
def main():
    # Configuración inicial
    check_dependencies()  # <-- Verifica dependencias primero
    st.set_page_config(page_title="GA4 Explorer", layout="wide")
    st.title("📊 Análisis Exploratorio GA4")

    # --- Sidebar: Configuración ---
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

    # --- Conexión a BigQuery ---
    client = get_bq_client(
        st.session_state.creds if development_mode and "creds" in st.session_state else None
    )

    # --- Selectores de proyecto/dataset ---
    try:
        projects = [p.project_id for p in client.list_projects()]
        selected_project = st.sidebar.selectbox("Proyecto", projects, key="project_select")
        datasets = [d.dataset_id for d in client.list_datasets(selected_project)]
        selected_dataset = st.sidebar.selectbox("Dataset GA4", datasets, key="dataset_select")
        
        # Guarda en sesión
        st.session_state.client = client
        st.session_state.project = selected_project
        st.session_state.dataset = selected_dataset
    except Exception as e:
        st.error(f"🗄️ Error al listar recursos:\n{str(e)}")
        st.stop()

    # --- Consulta principal ---
    st.header("🔍 Consulta básica")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Fecha inicio", value=pd.to_datetime("2023-01-01"), key="start_date")
    with col2:
        end_date = st.date_input("Fecha fin", value=pd.to_datetime("today"), key="end_date")
    
    if st.button("Ejecutar consulta"):
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
        with st.expander("🔍 Metadatos"):
            st.write(f"📏 Filas: {len(df)}")
            st.write("📊 Tipos de datos:", df.dtypes)

if __name__ == "__main__":
    main()
