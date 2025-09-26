import streamlit as st
from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

def generar_query_ingresos_transacciones(project, dataset, start_date, end_date):
    """Consulta CORREGIDA para ingresos y transacciones"""
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    return f"""
    SELECT
        event_date AS date,
        COUNT(*) AS total_purchase_events,
        COUNT(DISTINCT ecommerce.transaction_id) AS unique_transactions,
        SUM(ecommerce.purchase_revenue) AS purchase_revenue,
        COUNT(DISTINCT user_pseudo_id) AS unique_buyers
    FROM
        `{project}.{dataset}.events_*`
    WHERE
        _TABLE_SUFFIX BETWEEN '{start_date_str}' AND '{end_date_str}'
        AND event_name = 'purchase'
        AND ecommerce.transaction_id IS NOT NULL
    GROUP BY 
        event_date
    ORDER BY 
        event_date
    """

def generar_query_productos_mas_vendidos(project, dataset, start_date, end_date):
    """NUEVA CONSULTA: Performance de productos por ingresos (CON NOMBRE)"""
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    return f"""
    WITH PurchaseItems AS (
        -- Extraer datos de items de compras con filtro de fecha
        SELECT
            items.item_id AS item_id,
            items.item_name AS item_name,
            items.quantity AS item_quantity,
            items.item_revenue AS item_revenue
        FROM
            `{project}.{dataset}.events_*`,
            UNNEST(items) AS items
        WHERE
            event_name = 'purchase' -- Filtrar eventos de compra
            AND _TABLE_SUFFIX BETWEEN '{start_date_str}' AND '{end_date_str}' -- Filtrado dinámico de fechas
    )
    SELECT
        item_id,
        item_name,
        SUM(item_quantity) AS total_quantity_sold, -- Cantidad total vendida
        COUNT(*) AS total_purchases, -- Número total de eventos de compra
        SUM(item_revenue) AS total_revenue -- Ingreso total del producto
    FROM 
        PurchaseItems
    GROUP BY 
        item_id, item_name
    ORDER BY 
        total_revenue DESC -- Ordenar por ingreso total descendente
    """

def generar_query_relacion_productos(project, dataset, start_date, end_date):
    """NUEVA CONSULTA: Relación ID vs Nombre de productos"""
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    return f"""
    WITH ProductEvents AS (
        SELECT DISTINCT
            items.item_id AS item_id,
            items.item_name AS item_name
        FROM
            `{project}.{dataset}.events_*`,
            UNNEST(items) AS items
        WHERE
            _TABLE_SUFFIX BETWEEN '{start_date_str}' AND '{end_date_str}'
            AND items.item_id IS NOT NULL
            AND items.item_name IS NOT NULL
    )
    SELECT
        item_id,
        item_name,
        COUNT(*) OVER(PARTITION BY item_name) AS nombres_por_producto,
        COUNT(*) OVER(PARTITION BY item_id) AS ids_por_nombre
    FROM 
        ProductEvents
    ORDER BY 
        item_name, item_id
    """
def generar_query_detalle_transacciones(project, dataset, start_date, end_date):
    """CONSULTA SIN LÍMITES: Todas las transacciones disponibles"""
    start_date_str = start_date.strftime('%Y%m%d')
    end_date_str = end_date.strftime('%Y%m%d')
    
    return f"""
    SELECT
        ep.value.int_value as transaction_id,
        COUNT(*) as total_eventos,
        SUM(ecommerce.purchase_revenue) as revenue_total,
        SUM(ecommerce.total_item_quantity) as total_items,
        COUNT(DISTINCT user_pseudo_id) as unique_buyers
    FROM
        `{project}.{dataset}.events_*`,
        UNNEST(event_params) as ep
    WHERE
        event_name = 'purchase'
        AND _TABLE_SUFFIX BETWEEN '{start_date_str}' AND '{end_date_str}'
        AND ep.key = 'transaction_id'
        AND ep.value.int_value IS NOT NULL
    GROUP BY
        ep.value.int_value
    HAVING
        SUM(ecommerce.purchase_revenue) > 0
    ORDER BY
        revenue_total DESC  -- Ordenar por ingresos para ver las más importantes primero
    -- SIN LIMIT: Procesa todas las transacciones
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
    st.metric("📉 Eventos sin consentimiento (Real)", f"{dened_pct:.2f}%")

def mostrar_comparativa_eventos(df):
    """Visualización para comparativa completa de eventos (con funnel como antes)"""
    st.subheader("📊 Funnel de Ecommerce")
    
    if df.empty:
        st.warning("No hay datos disponibles para el rango seleccionado")
        return
    
    # Mostrar tabla con datos crudos
    st.dataframe(df.style.format({
        'total_events': '{:,}',
        'unique_users': '{:,}'
    }))
    
    # Agregar datos por tipo de evento (suma total)
    event_totals = df.groupby('event_name').agg({
        'total_events': 'sum',
        'unique_users': 'sum'
    }).reset_index()
    
    # Crear datos para el funnel - asegurarnos de que todos los eventos existan
    funnel_data = []
    event_types = ['page_view', 'view_item', 'add_to_cart', 'begin_checkout', 'purchase']
    
    for event_type in event_types:
        event_data = event_totals[event_totals['event_name'] == event_type]
        if not event_data.empty:
            funnel_data.append({
                'event_name': event_type,
                'total_events': event_data['total_events'].values[0],
                'unique_users': event_data['unique_users'].values[0]
            })
        else:
            # Si no hay datos para este evento, agregar cero
            funnel_data.append({
                'event_name': event_type,
                'total_events': 0,
                'unique_users': 0
            })
    
    funnel_df = pd.DataFrame(funnel_data)
    
    # Calcular tasas de conversión con manejo de zeros
    page_views = funnel_df[funnel_df['event_name'] == 'page_view']['total_events'].values[0]
    view_items = funnel_df[funnel_df['event_name'] == 'view_item']['total_events'].values[0]
    add_to_cart = funnel_df[funnel_df['event_name'] == 'add_to_cart']['total_events'].values[0]
    begin_checkout = funnel_df[funnel_df['event_name'] == 'begin_checkout']['total_events'].values[0]
    purchases = funnel_df[funnel_df['event_name'] == 'purchase']['total_events'].values[0]
    
    # Mostrar métricas de conversión con manejo de división por cero
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
    
    # Gráfico de funnel (solo mostrar eventos con datos)
    funnel_events = []
    funnel_values = []
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", '#d62728', '#9467bd']
    event_labels = {
        'page_view': 'Page Views',
        'view_item': 'View Item', 
        'add_to_cart': 'Add to Cart',
        'begin_checkout': 'Begin Checkout',
        'purchase': 'Purchase'
    }
    
    for i, event_type in enumerate(event_types):
        event_value = funnel_df[funnel_df['event_name'] == event_type]['total_events'].values[0]
        if event_value > 0:  # Solo agregar eventos con datos
            funnel_events.append(event_labels[event_type])
            funnel_values.append(event_value)
    
    if funnel_values:  # Solo crear gráfico si hay datos
        fig_funnel = go.Figure(go.Funnel(
            y=funnel_events,
            x=funnel_values,
            textinfo="value+percent initial",
            opacity=0.8,
            marker={"color": colors[:len(funnel_events)]}
        ))
        
        fig_funnel.update_layout(title="Funnel de Conversión de Ecommerce")
        st.plotly_chart(fig_funnel, use_container_width=True)

def mostrar_ingresos_transacciones(df):
    """Visualización CORREGIDA para ingresos y transacciones"""
    st.subheader("💰 Ingresos y Transacciones")
    
    if df.empty:
        st.warning("No hay datos de transacciones para el rango seleccionado")
        return
    
    # Convertir la fecha a formato legible
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df['fecha_formateada'] = df['date'].dt.strftime('%d/%m/%Y')
    
    # Mostrar tabla con datos crudos
    st.dataframe(df.style.format({
        'total_purchase_events': '{:,}',
        'unique_transactions': '{:,}',
        'purchase_revenue': '€{:,.2f}',
        'unique_buyers': '{:,}'
    }))
    
    # Calcular métricas totales
    total_purchases = df['total_purchase_events'].sum()
    total_revenue = df['purchase_revenue'].sum()
    avg_transaction_value = total_revenue / total_purchases if total_purchases > 0 else 0
    
    # Mostrar métricas clave
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Compras", f"{total_purchases:,}")
    with col2:
        st.metric("Ingresos Totales", f"€{total_revenue:,.2f}")
    with col3:
        st.metric("Ticket Medio", f"€{avg_transaction_value:,.2f}")
    
    # Gráfico combinado (ingresos + transacciones) - SOLO UN GRÁFICO
    fig = go.Figure()
    
    # Añadir ingresos (línea, eje izquierdo)
    fig.add_trace(go.Scatter(
        x=df['fecha_formateada'], 
        y=df['purchase_revenue'],
        name='Ingresos (€)',
        line=dict(color='#4CAF50', width=3),
        yaxis='y'
    ))
    
    # Añadir compras (barras, eje derecho) - Usamos total_purchase_events como proxy
    fig.add_trace(go.Bar(
        x=df['fecha_formateada'],
        y=df['total_purchase_events'],
        name='Compras',
        marker_color='#2196F3',
        opacity=0.6,
        yaxis='y2'
    ))
    
    # Configurar layout con doble eje Y
    fig.update_layout(
        title='Ingresos vs Compras',
        xaxis=dict(
            title='Fecha',
            tickangle=45,
            tickmode='array',
            tickvals=df['fecha_formateada'],
            ticktext=df['fecha_formateada']
        ),
        yaxis=dict(
            title='Ingresos (€)',
            titlefont=dict(color='#4CAF50'),
            tickfont=dict(color='#4CAF50'),
            side='left'
        ),
        yaxis2=dict(
            title='Compras',
            titlefont=dict(color='#2196F3'),
            tickfont=dict(color='#2196F3'),
            anchor='x',
            overlaying='y',
            side='right'
        ),
        legend=dict(x=0.02, y=0.98),
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def mostrar_productos_mas_vendidos(df):
    """NUEVA VISUALIZACIÓN: Performance de productos por ingresos (CON NOMBRE)"""
    st.subheader("🏆 Productos Más Vendidos por Ingresos")
    
    if df.empty:
        st.warning("No hay datos de productos vendidos para el rango seleccionado")
        return
    
    # Mostrar tabla con datos crudos
    st.dataframe(df.style.format({
        'total_quantity_sold': '{:,}',
        'total_purchases': '{:,}',
        'total_revenue': '€{:,.2f}'
    }))
    
    # Calcular métricas generales
    total_revenue = df['total_revenue'].sum()
    total_quantity = df['total_quantity_sold'].sum()
    avg_revenue_per_product = df['total_revenue'].mean()
    
    # Mostrar métricas clave
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Ingresos Totales Productos", f"€{total_revenue:,.2f}")
    with col2:
        st.metric("Cantidad Total Vendida", f"{total_quantity:,}")
    with col3:
        st.metric("Ingreso Promedio por Producto", f"€{avg_revenue_per_product:,.2f}")
    
    # Gráfico de barras - Top 10 productos por ingresos (USANDO NOMBRE)
    top_products = df.head(10).copy()
    
    # Crear gráfico de barras horizontal para mejor visualización
    fig_bar = px.bar(
        top_products,
        y='item_name',
        x='total_revenue',
        orientation='h',
        title='Top 10 Productos por Ingresos',
        labels={'total_revenue': 'Ingresos (€)', 'item_name': 'Nombre del Producto'},
        color='total_revenue',
        color_continuous_scale='Viridis',
        hover_data=['item_id']  # Mostrar ID en hover
    )
    
    fig_bar.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=500
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Gráfico de dispersión: Cantidad vs Ingresos (USANDO NOMBRE)
    fig_scatter = px.scatter(
        df.head(20),  # Limitar a top 20 para mejor visualización
        x='total_quantity_sold',
        y='total_revenue',
        size='total_purchases',
        color='total_revenue',
        hover_name='item_name',
        hover_data=['item_id'],
        title='Relación: Cantidad Vendida vs Ingresos',
        labels={
            'total_quantity_sold': 'Cantidad Total Vendida',
            'total_revenue': 'Ingresos Totales (€)',
            'total_purchases': 'Número de Compras'
        },
        size_max=30
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

def mostrar_relacion_productos(df):
    """NUEVA VISUALIZACIÓN: Relación ID vs Nombre de productos"""
    st.subheader("🔍 Relación ID vs Nombre de Productos")
    
    if df.empty:
        st.warning("No hay datos de productos para el rango seleccionado")
        return
    
    # Identificar ineficiencias
    df_ineficiencias = df[
        (df['nombres_por_producto'] > 1) | (df['ids_por_nombre'] > 1)
    ].copy()
    
    # Aplicar formato condicional para resaltar ineficiencias
    def highlight_inefficiencies(row):
        styles = [''] * len(row)
        if row['nombres_por_producto'] > 1 or row['ids_por_nombre'] > 1:
            styles = ['background-color: #ffcccc' for _ in row]  # Rojo claro para ineficiencias
        return styles
    
    # Mostrar tabla completa con resaltado
    st.write("**Tabla completa de relación ID vs Nombre:**")
    styled_df = df.style.format({
        'nombres_por_producto': '{:,}',
        'ids_por_nombre': '{:,}'
    }).apply(highlight_inefficiencies, axis=1)
    
    st.dataframe(styled_df)
    
    # Resumen de ineficiencias
    if not df_ineficiencias.empty:
        st.warning("🚨 **Se detectaron posibles ineficiencias:**")
        
        # Productos con múltiples nombres
        productos_multi_nombre = df_ineficiencias[df_ineficiencias['nombres_por_producto'] > 1]
        if not productos_multi_nombre.empty:
            st.write("**Productos con múltiples nombres:**")
            for _, row in productos_multi_nombre.head(5).iterrows():
                st.write(f"- ID `{row['item_id']}` tiene {int(row['nombres_por_producto'])} nombres diferentes")
        
        # Nombres con múltiples IDs
        nombres_multi_id = df_ineficiencias[df_ineficiencias['ids_por_nombre'] > 1]
        if not nombres_multi_id.empty:
            st.write("**Nombres con múltiples IDs:**")
            for _, row in nombres_multi_id.head(5).iterrows():
                st.write(f"- Nombre `{row['item_name']}` tiene {int(row['ids_por_nombre'])} IDs diferentes")
        
        # Métricas de ineficiencia
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Productos con ineficiencias", len(df_ineficiencias))
        with col2:
            st.metric("% de ineficiencia", f"{(len(df_ineficiencias) / len(df) * 100):.1f}%")
    else:
        st.success("✅ No se detectaron ineficiencias en la relación ID vs Nombre")

def mostrar_detalle_transacciones(df):
    """VISUALIZACIÓN SIMPLIFICADA: Sin filtros ni muestreo"""
    st.subheader("🧾 Detalle Completo de Transacciones")
    
    if df.empty:
        st.warning("No hay datos de transacciones para el rango seleccionado")
        return
    
    # Informar sobre el volumen de datos
    total_transacciones = len(df)
    st.success(f"📈 Se encontraron {total_transacciones:,} transacciones en total")
    
    # Convertir transaction_id a string para mejor visualización
    df['transaction_id'] = df['transaction_id'].astype(str)
    
    # Mostrar métricas generales
    total_revenue = df['revenue_total'].sum()
    avg_transaction_value = df['revenue_total'].mean()
    total_items = df['total_items'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Transacciones", f"{total_transacciones:,}")
    with col2:
        st.metric("Ingresos Totales", f"€{total_revenue:,.2f}")
    with col3:
        st.metric("Ticket Medio", f"€{avg_transaction_value:,.2f}")
    with col4:
        st.metric("Items Totales", f"{total_items:,}")
    
    # Mostrar tabla con TODOS los datos
    st.dataframe(df.style.format({
        'transaction_id': '{}',
        'total_eventos': '{:,}',
        'revenue_total': '€{:,.2f}',
        'total_items': '{:,}',
        'unique_buyers': '{:,}'
    }), height=400)
    
    # Gráficos de análisis con todos los datos
    if not df.empty:
        tab1, tab2 = st.tabs(["Distribución de Ingresos", "Relación Items vs Ingresos"])
        
        with tab1:
            fig_hist = px.histogram(df, 
                                   x='revenue_total',
                                   nbins=min(30, len(df)//10),
                                   title='Distribución de Ingresos por Transacción',
                                   labels={'revenue_total': 'Ingresos (€)', 'count': 'Número de Transacciones'})
            fig_hist.update_layout(bargap=0.1)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab2:
            fig_scatter = px.scatter(df,
                                    x='total_items',
                                    y='revenue_total',
                                    size='unique_buyers',
                                    color='revenue_total',
                                    hover_name='transaction_id',
                                    title='Relación: Cantidad de Items vs Ingresos',
                                    labels={
                                        'total_items': 'Cantidad de Items',
                                        'revenue_total': 'Ingresos (€)',
                                        'unique_buyers': 'Compradores Únicos'
                                    },
                                    color_continuous_scale='viridis')
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Análisis de estadísticas
    st.subheader("📊 Estadísticas del Conjunto de Datos")
    
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            max_transaction = df['revenue_total'].max()
            st.metric("Transacción Más Alta", f"€{max_transaction:,.2f}")
        
        with col2:
            min_transaction = df['revenue_total'].min()
            st.metric("Transacción Más Baja", f"€{min_transaction:,.2f}")
        
        with col3:
            median_transaction = df['revenue_total'].median()
            st.metric("Mediana", f"€{median_transaction:,.2f}")
        
        with col4:
            std_transaction = df['revenue_total'].std()
            st.metric("Desviación Estándar", f"€{std_transaction:,.2f}")
        
        # Resumen de percentiles
        st.write("**Distribución por Percentiles:**")
        percentiles = df['revenue_total'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("P10", f"€{percentiles[0.1]:,.2f}")
        with col2:
            st.metric("P25", f"€{percentiles[0.25]:,.2f}")
        with col3:
            st.metric("P50 (Mediana)", f"€{percentiles[0.5]:,.2f}")
        with col4:
            st.metric("P75", f"€{percentiles[0.75]:,.2f}")
        with col5:
            st.metric("P90", f"€{percentiles[0.9]:,.2f}")
        
        # Top 10 transacciones
        st.subheader("🏆 Top 10 Transacciones por Ingresos")
        top10 = df.head(10).copy()
        
        for idx, row in top10.iterrows():
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.write(f"**Transacción ID:** {row['transaction_id']}")
                with col2:
                    st.metric("Ingresos", f"€{row['revenue_total']:,.2f}")
                with col3:
                    st.metric("Items", f"{row['total_items']:,}")
                with col4:
                    st.metric("Compradores", f"{row['unique_buyers']:,}")
                st.divider()

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
        if st.button("Ejecutar Análisis de Funnel", key="btn_funnel"):
            with st.spinner("Analizando funnel de conversión..."):
                query = generar_query_comparativa_eventos(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_comparativa_eventos(df)
    
    with st.expander("💰 Ingresos y Transacciones", expanded=True):
        if st.button("Analizar Ingresos y Transacciones", key="btn_ingresos"):
            with st.spinner("Calculando ingresos y transacciones..."):
                query = generar_query_ingresos_transacciones(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_ingresos_transacciones(df)
    
    # NUEVA SECCIÓN: Detalle de Transacciones (SIMPLIFICADA)
    with st.expander("🧾 Detalle de Transacciones", expanded=True):
        if st.button("Analizar Detalle de Transacciones", key="btn_detalle_transacciones"):
            with st.spinner("Obteniendo detalle de transacciones..."):
                query = generar_query_detalle_transacciones(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_detalle_transacciones(df)
    
    with st.expander("🏆 Productos Más Vendidos", expanded=True):
        if st.button("Analizar Performance de Productos", key="btn_productos"):
            with st.spinner("Analizando productos más vendidos..."):
                query = generar_query_productos_mas_vendidos(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_productos_mas_vendidos(df)
    
    with st.expander("🔍 Relación ID vs Nombre de Productos", expanded=True):
        if st.button("Analizar Relación Productos", key="btn_relacion"):
            with st.spinner("Analizando relación ID vs Nombre..."):
                query = generar_query_relacion_productos(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_relacion_productos(df)
    
    with st.expander("📈 Funnel de Conversión por Producto", expanded=True):
        if st.button("Analizar Funnel por Producto", key="btn_funnel_producto"):
            with st.spinner("Analizando funnel por producto..."):
                query = generar_query_funnel_por_producto(project, dataset, start_date, end_date)
                df = run_query(client, query)
                mostrar_funnel_por_producto(df)

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
