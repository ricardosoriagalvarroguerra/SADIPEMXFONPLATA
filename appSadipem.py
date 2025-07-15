import streamlit as st
import pandas as pd
import os
from PIL import Image  # Agregar importación para imágenes

# Ruta al archivo Parquet
DATA_PATH = os.path.join(os.path.dirname(__file__), 'bdd_sadipem_10.parquet')

# Cargar datos
@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

data = load_data()
# Reemplazar nombre largo por 'BNDS' directamente en la base
data['nombre_acreedor'] = data['nombre_acreedor'].replace('Banco Nacional de Desenvolvimento Econômico e Social', 'BNDS')
# Reemplazar 'New Development Bank' por 'NDB'
data['nombre_acreedor'] = data['nombre_acreedor'].replace('New Development Bank', 'NDB')
# Reemplazar 'Other' por 'No Clasificado' en sector
data['sector'] = data['sector'].replace('Other', 'No Clasificado')

# Cargar imagen de logo para los títulos
LOGO_PATH = os.path.join(os.path.dirname(__file__), 'startfonp.png')
logo_rel_path = 'startfonp.png'  # Para usar en HTML
logo_img = Image.open(LOGO_PATH)

# Sidebar para navegación
st.sidebar.title('Menú de Navegación')
pages = [
    'Home',
    'Metodologia',
    'Interno Vs Externo',
    'Regiones por Financiador',
    'Regiones por Sector',
    'Financiador por Sector',
    'Dispersion',
    'Box Plots'
]
page = st.sidebar.selectbox('Selecciona la página:', pages)

# Renderizado de páginas
if page == 'Home':
    header_cols = st.columns([0.12, 0.88])
    with header_cols[0]:
        st.image(logo_img, width=60)
    with header_cols[1]:
        st.markdown('<h1 style="margin:0; padding:0;">FONPLATA - SADIPEM</h1>', unsafe_allow_html=True)
    st.write('Bienvenido a la app de análisis de la BDD SADIPEM.')
    st.write('Utiliza el menú lateral para navegar entre las diferentes páginas.')
    st.markdown('---')
    st.markdown('<h3 style="margin-top:0;">Descripción de las secciones</h3>', unsafe_allow_html=True)
    section_info = [
        ("Home", "Página de inicio con información general sobre la aplicación y su navegación."),
        ("Interno Vs Externo", "Comparación temporal de los financiamientos internos y externos, para identificar tendencias y diferencias entre ambos tipos de recursos."),
        ("Regiones por Financiador", "Distribución de los financiamientos por acreedor en las distintas regiones, facilitando el análisis de la presencia regional de cada financiador."),
        ("Regiones por Sector", "Visualización de los sectores que reciben financiamiento en cada región, útil para identificar prioridades sectoriales a nivel regional."),
        ("Financiador por Sector", "Sectores financiados por cada acreedor principal, proporcionando una visión clara del enfoque sectorial de cada financiador."),
        ("Dispersión", "Relación entre el valor de los préstamos y el plazo de repago, diferenciando por financiador, para detectar patrones y casos atípicos."),
        ("Box Plots", "Comparación de la distribución de los montos financiados entre los principales acreedores mediante diagramas de caja, mostrando rangos, medianas y valores atípicos.")
    ]
    cols = st.columns(2)
    for idx, (title, desc) in enumerate(section_info):
        with cols[idx % 2]:
            st.markdown(f'''
                <div style="background: #fff; border: 1.5px solid #c1121f; border-radius: 12px; padding: 18px 20px; margin-bottom: 18px; min-height: 120px; box-shadow: 0 2px 8px rgba(193,18,31,0.04);">
                    <h4 style="margin-top:0; margin-bottom:8px; color:#c1121f;">{title}</h4>
                    <div style="color:#222; font-size:1.07em;">{desc}</div>
                </div>
            ''', unsafe_allow_html=True)
    st.markdown('---')
    st.markdown('<p style="font-size:1.04em; margin-top:18px; margin-bottom:0;"><b>Sobre SADIPEM:</b> El Sistema de Análisis de la Deuda Pública y Endeudamiento Municipal (SADIPEM) es una base de datos desarrollada para el monitoreo y análisis del financiamiento subnacional en Brasil. SADIPEM se alimenta de registros oficiales provenientes de organismos multilaterales, bancos de desarrollo y entidades gubernamentales.</p>', unsafe_allow_html=True)
elif page == 'Metodologia':
    header_cols = st.columns([0.12, 0.88])
    with header_cols[0]:
        st.image(logo_img, width=60)
    with header_cols[1]:
        st.markdown('<h1 style="margin:0; padding:0;">Metodología de depuración y estandarización de datos</h1>', unsafe_allow_html=True)
    st.markdown('---')
    pasos = [
        ("Exploración inicial y verificación de encabezados", "Antes de cargar todo el dataset se abrió el archivo en modo texto y se leyó únicamente la primera línea para listar las columnas. Este paso sirvió para confirmar la codificación Latin-1, el separador ';' y la ortografía de cada encabezado; así se evitó que errores de tipeo pasaran inadvertidos en los filtros posteriores."),
        ("Carga del CSV original y filtrado por tipo de deuda", "Con pandas.read_csv() se leyó dividas.csv, habilitando la opción on_bad_lines='warn' para que las filas corruptas no interrumpieran la lectura. Inmediatamente se retuvo sólo la categoría 'Empréstimo ou financiamento' del campo Tipo de dívida. El resultado se guardó como dividas_1.csv, instalando desde el inicio un subconjunto homogéneo (sólo préstamos y financiamientos)."),
        ("Depuración de la columna 'Classificação no RGF'", "Sobre dividas_1.csv se aplicó un segundo filtro que conserva: 'Empréstimos internos', 'Empréstimos externos', 'Financiamientos internos', 'Financiamientos externos' y valores vacíos. Todo lo demás se descartó. El archivo intermedio quedó como dividas_2.csv."),
        ("Filtro por tipo de acreedor", "El tercer corte retuvo únicamente los registros cuyo Tipo de credor se encuentra entre: União, Instituição Financeira Nacional, Instituição Financeira Internacional. Así se aislaron los acreedores realmente relevantes y se escribió dividas_3.csv."),
        ("Estandarización y filtrado de la fecha de contratación", "La columna Data da contratação, emissão ou assunção se convirtió a datetime usando el patrón '%d/%m/%Y'. Después se restringieron las observaciones al rango 1-ene-2010 / 31-dic-2024. Las filas con fechas inválidas (NaT) quedaron fuera y el resultado se guardó en dividas_3_1.csv y luego, tras el recorte temporal, en dividas_3_2.csv."),
        ("Imputación automática de la 'Classificação no RGF' faltante", "Se reconstruyó la clasificación ausente basándose en Tipo de credor: Si el acreedor es União o Instituição Financeira Nacional, la fila se marca como 'Empréstimos internos'. Si el acreedor es Instituição Financeira Internacional, se marca como 'Empréstimos externos'. El DataFrame actualizado se guardó como dividas_3_2_clasif.csv."),
        ("Validación cruzada acreedor-clasificación y correcciones", "Se ejecutaron pruebas lógicas para detectar combinaciones imposibles (por ejemplo, un préstamo 'externo' con acreedor 'União'). Las incongruencias se corrigieron in situ y se produjo dividas_3_2_clasif_corregido.csv, asegurando que las reglas internas/externas y el tipo de acreedor ya no se contradigan."),
        ("Normalización de nombres de acreedores", "Para la columna Nome do credor se generó nome_normalizado, aplicando: lowercase y strip(), eliminación de sufijos societarios ('S/A', 'S.A.', 'LTDA', 'AG.', etc.). Con ello se consolidan distintas versiones de un mismo banco en una sola etiqueta, simplificando cualquier conteo o agrupación posterior."),
        ("Conversión de montos a numérico", "El campo monetario original — Valor da contratação, emissão ou assunção (na moeda de contratação) — llegaba como string con '.' para miles y ',' para decimales. Se creó la columna numérica _num eliminando los separadores de miles, reemplazando la coma por punto y usando pd.to_numeric(..., errors='coerce'). La base con el cambio se denominó dividas_4.csv."),
        ("Conversión de la fecha de cancelación", "Data da quitação se transformó a formato datetime y la tabla resultante se volcó como dividas_5.csv."),
        ("Análisis de valores atípicos", "Sobre dividas_4.csv se calcularon estadísticos descriptivos y límites IQR para _num. Se identificaron outliers globales y, en un paso aparte, se repitió el análisis separando Empréstimos internos y externos a fin de comparar distribuciones. Los outliers se almacenaron en outliers_valor_contratacao.csv para revisión manual."),
        ("Traducción de la descripción/finalidad", "La columna Descrição / finalidade se tradujo automáticamente al inglés con googletrans 4.0-rc1, creando Description_en. El archivo final multilingüe se exportó como dividas_6_traducida.csv en UTF-8."),
        ("Ajustes finales de nomenclatura", "En una fase posterior (fuera de la familia dividas pero dentro del mismo notebook) se renombró la columna tipo_garantia a garantia_soberana en otro DataFrame (bdd_sadipem_7.csv), manteniendo la coherencia terminológica entre bases conectadas."),
        ("Control de versiones intermedias", "Cada etapa crítica terminó con un to_csv, lo que permite regresar a cualquier corte del proceso sin re-ejecutar todo el notebook. Los archivos generados —dividas_1.csv, …, dividas_6_traducida.csv— actúan como puntos de restauración y evidencia de la trazabilidad de los cambios.")
    ]
    for idx, (title, desc) in enumerate(pasos):
        color = "#f8d7da" if idx % 2 == 0 else "#f1f3f4"
        st.markdown(f'''
            <div style="background: {color}; border-radius: 12px; padding: 18px 22px; margin-bottom: 18px; border-left: 5px solid #c1121f;">
                <h4 style="margin-top:0; margin-bottom:8px; color:#c1121f;">Paso {idx+1}: {title}</h4>
                <div style="color:#222; font-size:1.07em;">{desc}</div>
            </div>
        ''', unsafe_allow_html=True)
    st.markdown('<div style="margin-top:24px; font-size:1.08em;"><b>Resultado:</b> Mediante esta cadena de filtros, transformaciones, validaciones y enriquecimiento lingüístico se obtuvo un dataset coherente, libre de duplicidades conceptuales, con fechas y montos en formatos analíticos y una clasificación interno/externo consistente con el tipo de acreedor. Toda la operación quedó documentada paso a paso dentro del cuaderno y en los archivos intermedios, garantizando transparencia y reproducibilidad.</div>', unsafe_allow_html=True)
elif page == 'Interno Vs Externo':
    header_cols = st.columns([0.12, 0.88])
    with header_cols[0]:
        st.image(logo_img, width=60)
    with header_cols[1]:
        st.markdown('<h1 style="margin:0; padding:0;">Interno Vs Externo</h1>', unsafe_allow_html=True)
    st.write('Visualización comparativa entre Interno y Externo.')
    df = data.copy()
    # --- FILTROS EN SIDEBAR ---
    st.sidebar.markdown('---')
    st.sidebar.subheader('Filtros')
    # Filtro Garantía Soberana
    garantia_opts = ['Todos', 'Si', 'No']
    garantia_sel = st.sidebar.selectbox('Garantía Soberana', garantia_opts, index=0)
    if garantia_sel != 'Todos':
        df = df[df['garantia_soberana'] == garantia_sel]
    # Filtros selectbox primero
    tipo_entes = ['Todas'] + sorted(df['tipo_ente'].dropna().unique().tolist())
    tipo_ente_sel = st.sidebar.selectbox('Filtrar por tipo de ente', tipo_entes, index=0)
    if tipo_ente_sel != 'Todas':
        df = df[df['tipo_ente'] == tipo_ente_sel]
    sectores = ['Todas'] + sorted(df['sector'].dropna().unique().tolist())
    sector_sel = st.sidebar.selectbox('Filtrar por sector', sectores, index=0)
    regiones = ['Todas'] + sorted(df['região'].dropna().unique().tolist())
    region_sel = st.sidebar.selectbox('Filtrar por región', regiones, index=0)
    # Sliders después
    df['fecha_contratacion'] = pd.to_datetime(df['fecha_contratacion'], errors='coerce')
    df['año'] = df['fecha_contratacion'].dt.year
    min_year = int(df['fecha_contratacion'].dt.year.min())
    max_year = int(df['fecha_contratacion'].dt.year.max())
    year_range = st.sidebar.slider('Filtrar por año de contratación', min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1)
    min_usd_m = float(df['valor_usd'].min()) / 1_000_000
    max_usd_m = float(df['valor_usd'].max()) / 1_000_000
    valor_usd_range_m = st.sidebar.slider('Filtrar por valor USD (millones)', min_value=min_usd_m, max_value=max_usd_m, value=(min_usd_m, max_usd_m), step=0.1, format='%.2f')
    min_tiempo = float(df['tiempo_prestamo'].min())
    max_tiempo = float(df['tiempo_prestamo'].max())
    tiempo_prestamo_range = st.sidebar.slider('Filtrar por tiempo de préstamo (años)', min_value=min_tiempo, max_value=max_tiempo, value=(min_tiempo, max_tiempo), step=1.0, format='%.1f')
    # Aplicar filtros
    df = df[(df['fecha_contratacion'].dt.year >= year_range[0]) & (df['fecha_contratacion'].dt.year <= year_range[1])]
    df = df[(df['valor_usd'] >= valor_usd_range_m[0] * 1_000_000) & (df['valor_usd'] <= valor_usd_range_m[1] * 1_000_000)]
    df = df[(df['tiempo_prestamo'] >= tiempo_prestamo_range[0]) & (df['tiempo_prestamo'] <= tiempo_prestamo_range[1])]
    if sector_sel != 'Todas':
        df = df[df['sector'] == sector_sel]
    if region_sel != 'Todas':
        df = df[df['região'] == region_sel]
    # Agrupar por año y clasificación
    grouped = df.groupby(['año', 'RGF_clasificacion'])['valor_usd'].sum().reset_index()
    grouped['valor_usd_millones'] = grouped['valor_usd'] / 1_000_000
    # Definir el orden de las categorías: Interno abajo, Externo arriba
    import pandas as pd
    cat_type = pd.CategoricalDtype(['Interno', 'Externo'], ordered=True)
    grouped['RGF_clasificacion'] = grouped['RGF_clasificacion'].astype(cat_type)
    # Definir colores: Externo = #c1121f, Interno = #adb5bd
    color_map = {'Externo': '#c1121f', 'Interno': '#adb5bd'}
    # Crear gráfico stacked
    import plotly.express as px
    fig = px.bar(
        grouped.sort_values(['año', 'RGF_clasificacion']),
        x='año',
        y='valor_usd_millones',
        color='RGF_clasificacion',
        color_discrete_map=color_map,
        category_orders={'RGF_clasificacion': ['Interno', 'Externo']},
        labels={'valor_usd_millones': 'Valor USD acumulado (millones)', 'RGF_clasificacion': 'Clasificación'},  # Eliminar 'año' del label
        barmode='stack',
        title='Aprobaciones anuales (USD Millones)',
        height=350
    )
    fig.update_layout(title_x=0.5)
    fig.update_yaxes(showgrid=False)
    fig.update_xaxes(showgrid=False, title_text='')  # Quitar etiqueta eje x
    fig.update_traces(marker_line_color='black', marker_line_width=2)
    st.plotly_chart(fig, use_container_width=True)

    # Gráfico 100% stacked (proporcional)
    grouped_pct = grouped.copy()
    total_por_año = grouped_pct.groupby('año')['valor_usd_millones'].transform('sum')
    grouped_pct['porcentaje'] = grouped_pct['valor_usd_millones'] / total_por_año * 100
    fig_pct = px.bar(
        grouped_pct.sort_values(['año', 'RGF_clasificacion']),
        x='año',
        y='porcentaje',
        color='RGF_clasificacion',
        color_discrete_map=color_map,
        category_orders={'RGF_clasificacion': ['Interno', 'Externo']},
        labels={'porcentaje': 'Porcentaje (%)', 'año': 'Año', 'RGF_clasificacion': 'Clasificación'},
        barmode='stack',
        title='%',
        height=300
    )
    fig_pct.update_layout(title_x=0.5)
    fig_pct.update_yaxes(range=[0, 100], showgrid=False)
    fig_pct.update_xaxes(showgrid=False)  # Mantener etiqueta eje x
    fig_pct.update_traces(marker_line_color='black', marker_line_width=2)
    st.plotly_chart(fig_pct, use_container_width=True)

elif page == 'Regiones por Financiador':
    header_cols = st.columns([0.12, 0.88])
    with header_cols[0]:
        st.image(logo_img, width=60)
    with header_cols[1]:
        st.markdown('<h1 style="margin:0; padding:0;">Regiones por Financiador</h1>', unsafe_allow_html=True)
    st.write('Análisis de regiones según el financiador.')
    df = data.copy()
    # --- FILTROS EN SIDEBAR ---
    st.sidebar.markdown('---')
    st.sidebar.subheader('Filtros')
    # Filtro Garantía Soberana
    garantia_opts = ['Todos', 'Si', 'No']
    garantia_sel = st.sidebar.selectbox('Garantía Soberana', garantia_opts, index=0)
    if garantia_sel != 'Todos':
        df = df[df['garantia_soberana'] == garantia_sel]
    # Filtros selectbox primero
    tipo_entes = ['Todas'] + sorted(df['tipo_ente'].dropna().unique().tolist())
    tipo_ente_sel = st.sidebar.selectbox('Filtrar por tipo de ente', tipo_entes, index=0)
    if tipo_ente_sel != 'Todas':
        df = df[df['tipo_ente'] == tipo_ente_sel]
    sectores = ['Todas'] + sorted(df['sector'].dropna().unique().tolist())
    sector_sel = st.sidebar.selectbox('Filtrar por sector', sectores, index=0)
    tipos_fin = ['Todos'] + sorted(df['RGF_clasificacion'].dropna().unique().tolist())
    tipo_fin_sel = st.sidebar.selectbox('Tipo de financiamiento', tipos_fin, index=0)
    # Sliders después
    df['fecha_contratacion'] = pd.to_datetime(df['fecha_contratacion'], errors='coerce')
    df['año'] = df['fecha_contratacion'].dt.year
    min_year = int(df['fecha_contratacion'].dt.year.min())
    max_year = int(df['fecha_contratacion'].dt.year.max())
    year_range = st.sidebar.slider('Filtrar por año de contratación', min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1)
    min_usd_m = float(df['valor_usd'].min()) / 1_000_000
    max_usd_m = float(df['valor_usd'].max()) / 1_000_000
    valor_usd_range_m = st.sidebar.slider('Filtrar por valor USD (millones)', min_value=min_usd_m, max_value=max_usd_m, value=(min_usd_m, max_usd_m), step=0.1, format='%.2f')
    min_tiempo = float(df['tiempo_prestamo'].min())
    max_tiempo = float(df['tiempo_prestamo'].max())
    tiempo_prestamo_range = st.sidebar.slider('Filtrar por tiempo de préstamo (años)', min_value=min_tiempo, max_value=max_tiempo, value=(min_tiempo, max_tiempo), step=1.0, format='%.1f')
    # Aplicar filtros
    df = df[(df['fecha_contratacion'].dt.year >= year_range[0]) & (df['fecha_contratacion'].dt.year <= year_range[1])]
    df = df[(df['valor_usd'] >= valor_usd_range_m[0] * 1_000_000) & (df['valor_usd'] <= valor_usd_range_m[1] * 1_000_000)]
    df = df[(df['tiempo_prestamo'] >= tiempo_prestamo_range[0]) & (df['tiempo_prestamo'] <= tiempo_prestamo_range[1])]
    if sector_sel != 'Todas':
        df = df[df['sector'] == sector_sel]
    if tipo_fin_sel != 'Todos':
        df = df[df['RGF_clasificacion'] == tipo_fin_sel]
    # --- GRAFICOS DE DONUTS EN SUBPLOTS ---
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    regiones_unicas = list(df['região'].dropna().unique())
    n = len(regiones_unicas)
    if n == 0:
        st.info('No hay datos para mostrar.')
    else:
        # --- LEYENDA PERSONALIZADA ---
        # Obtener todos los financiadores y colores presentes en los donuts
        from collections import OrderedDict
        color_map = {
            'BIRF': '#003049',
            'BID': '#034078',
            'FONPLATA': '#c1121f',
            'CAF': '#38b000',
            'Caixa': '#ffc600',
            'Unión': '#f18701',
            'BNDS': '#246a73',
            'Otros': '#b7bdc1',
            'Banco do Brasil S/A': '#2176ff',
            'Bank of America Merrill Lynch': '#fdf0d5',
        }
        # Recolectar financiadores presentes en los datos filtrados
        financiadores_presentes = OrderedDict()
        # Al recolectar financiadores presentes para la leyenda:
        for idx, reg in enumerate(regiones_unicas):
            df_reg = df[df['região'] == reg]
            top_acreedores = df_reg.groupby('nombre_acreedor')['valor_usd'].sum().nlargest(5)
            for nombre in top_acreedores.index:
                nombre_legenda = 'BNDS' if nombre == 'Banco Nacional de Desenvolvimento Econômico e Social' else nombre
                financiadores_presentes[nombre_legenda] = color_map.get(nombre_legenda, '#CCCCCC')
            if df_reg[~df_reg['nombre_acreedor'].isin(top_acreedores.index)].shape[0] > 0:
                financiadores_presentes['Otros'] = color_map.get('Otros', '#b7bdc1')
        # --- LEYENDA PERSONALIZADA VERTICAL ---
        # (Ya no se renderiza ninguna leyenda antes de los gráficos, solo en la segunda fila con los donuts)

        # --- GRAFICOS DE DONUTS Y LEYENDA EN LA SEGUNDA FILA ---
        cols = 3
        n = len(regiones_unicas)
        rows = (n + cols - 1) // cols
        # Renderizar primera fila de donuts
        if n > 0:
            donut_cols = st.columns(cols)
            for i in range(min(cols, n)):
                with donut_cols[i]:
                    reg = regiones_unicas[i]
                    df_reg = df[df['região'] == reg]
                    top_acreedores = df_reg.groupby('nombre_acreedor')['valor_usd'].sum().nlargest(5)
                    otros = df_reg[~df_reg['nombre_acreedor'].isin(top_acreedores.index)]['valor_usd'].sum()
                    labels = list(top_acreedores.index)
                    values = list(top_acreedores.values)
                    if otros > 0:
                        labels.append('Otros')
                        values.append(otros)
                    import plotly.express as px
                    default_colors = px.colors.qualitative.Plotly
                    def get_color(nombre):
                        if nombre in color_map:
                            return color_map[nombre]
                        else:
                            # Asignar color por orden alfabético para consistencia
                            all_labels = list(color_map.keys()) + sorted(set(financiadores_presentes.keys()) - set(color_map.keys()))
                            idx = all_labels.index(nombre)
                            return default_colors[idx % len(default_colors)]
                    def label_replace(nombre):
                        if nombre == 'Banco Nacional de Desenvolvimento Econômico e Social':
                            return 'BNDS'
                        return nombre
                    labels = [label_replace(n) for n in list(top_acreedores.index)]
                    values = list(top_acreedores.values)
                    if otros > 0:
                        labels.append('Otros')
                        values.append(otros)
                    colors = [get_color(label) for label in labels]
                    hovertexts = [f"{n}<br>{v/1_000_000:.2f} M USD" for n, v in zip(list(top_acreedores.index), list(top_acreedores.values))]
                    if otros > 0:
                        hovertexts.append(f'Otros<br>{otros/1_000_000:.2f} M USD')
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.75,
                            textinfo='percent',
                            textposition='inside',
                            showlegend=False,
                            marker=dict(colors=colors, line=dict(color='black', width=2)),
                            hovertext=hovertexts,
                            hoverinfo='text+percent',
                            domain={'x': [0.08, 0.92], 'y': [0.08, 0.92]}
                        )
                    ])
                    fig.update_traces(textfont_size=12, textinfo='percent')
                    # Forzar salto de línea para títulos cortos para mejor centrado visual
                    fig.update_layout(
                        margin=dict(t=0, b=0, l=0, r=0),
                        height=260,
                    )
                    # Detectar tema de Streamlit
                    theme_base = st.get_option('theme.base')
                    region_title_color = 'white' if theme_base == 'dark' else 'black'
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text=f"<b>{reg}</b>",
                        showarrow=False,
                        font=dict(size=22, color=region_title_color),
                        xref="paper",
                        yref="paper",
                        xanchor="center",
                        yanchor="middle",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        # Renderizar segunda fila: leyenda + donuts restantes
        if n > cols:
            donut_cols2 = st.columns(cols)
            # Leyenda en la primera columna
            with donut_cols2[0]:
                legend_html = "<div style='display: flex; flex-direction: column; gap: 8px; align-items: flex-start; margin-top: 10px; margin-bottom: 30px;'>"
                # Leyenda personalizada: usar color_map global para todos los financiadores
                def get_color_leyenda(nombre):
                    if nombre in color_map:
                        return color_map[nombre]
                    else:
                        # Asignar color por orden alfabético para consistencia
                        all_labels = list(color_map.keys()) + sorted(set(financiadores_presentes.keys()) - set(color_map.keys()))
                        idx = all_labels.index(nombre)
                        return default_colors[idx % len(default_colors)]
                # En la leyenda personalizada:
                for nombre, _ in financiadores_presentes.items():
                    color = get_color_leyenda(nombre)
                    legend_html += f"<div style='display: flex; align-items: center; gap: 6px;'><div style='width: 14px; height: 14px; background: {color}; border-radius: 3px; border: 1px solid #888;'></div><span style='font-size: 13px;'>{nombre}</span></div>"
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)
            # Donuts restantes en las otras columnas
            donuts_restantes = n - cols
            for j in range(cols, n):
                col_idx = j - cols + 1
                if donuts_restantes == 1 and col_idx == 2:
                    continue  # Deja la tercera columna vacía si solo hay dos donuts
                with donut_cols2[col_idx]:
                    reg = regiones_unicas[j]
                    df_reg = df[df['região'] == reg]
                    top_acreedores = df_reg.groupby('nombre_acreedor')['valor_usd'].sum().nlargest(5)
                    otros = df_reg[~df_reg['nombre_acreedor'].isin(top_acreedores.index)]['valor_usd'].sum()
                    labels = list(top_acreedores.index)
                    values = list(top_acreedores.values)
                    if otros > 0:
                        labels.append('Otros')
                        values.append(otros)
                    import plotly.express as px
                    default_colors = px.colors.qualitative.Plotly
                    def get_color(nombre):
                        if nombre in color_map:
                            return color_map[nombre]
                        else:
                            # Asignar color por orden alfabético para consistencia
                            all_labels = list(color_map.keys()) + sorted(set(financiadores_presentes.keys()) - set(color_map.keys()))
                            idx = all_labels.index(nombre)
                            return default_colors[idx % len(default_colors)]
                    def label_replace(nombre):
                        if nombre == 'Banco Nacional de Desenvolvimento Econômico e Social':
                            return 'BNDS'
                        return nombre
                    labels = [label_replace(n) for n in list(top_acreedores.index)]
                    values = list(top_acreedores.values)
                    if otros > 0:
                        labels.append('Otros')
                        values.append(otros)
                    colors = [get_color(label) for label in labels]
                    hovertexts = [f"{n}<br>{v/1_000_000:.2f} M USD" for n, v in zip(list(top_acreedores.index), list(top_acreedores.values))]
                    if otros > 0:
                        hovertexts.append(f'Otros<br>{otros/1_000_000:.2f} M USD')
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.75,
                            textinfo='percent',
                            textposition='inside',
                            showlegend=False,
                            marker=dict(colors=colors, line=dict(color='black', width=2)),
                            hovertext=hovertexts,
                            hoverinfo='text+percent',
                            domain={'x': [0.08, 0.92], 'y': [0.08, 0.92]}
                        )
                    ])
                    fig.update_traces(textfont_size=12, textinfo='percent')
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=260)
                    # Detectar tema de Streamlit
                    theme_base = st.get_option('theme.base')
                    region_title_color = 'white' if theme_base == 'dark' else 'black'
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text=f"<b>{reg}</b>",
                        showarrow=False,
                        font=dict(size=22, color=region_title_color),
                        xref="paper",
                        yref="paper",
                        xanchor="center",
                        yanchor="middle",
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif page == 'Regiones por Sector':
    header_cols = st.columns([0.12, 0.88])
    with header_cols[0]:
        st.image(logo_img, width=60)
    with header_cols[1]:
        st.markdown('<h1 style="margin:0; padding:0;">Regiones por Sector</h1>', unsafe_allow_html=True)
    st.write('Análisis de regiones según el sector.')
    df = data.copy()
    # --- FILTROS EN SIDEBAR ---
    st.sidebar.markdown('---')
    st.sidebar.subheader('Filtros')
    # Filtro Garantía Soberana
    garantia_opts = ['Todos', 'Si', 'No']
    garantia_sel = st.sidebar.selectbox('Garantía Soberana', garantia_opts, index=0)
    if garantia_sel != 'Todos':
        df = df[df['garantia_soberana'] == garantia_sel]
    # Filtros selectbox primero
    tipo_entes = ['Todas'] + sorted(df['tipo_ente'].dropna().unique().tolist())
    tipo_ente_sel = st.sidebar.selectbox('Filtrar por tipo de ente', tipo_entes, index=0)
    if tipo_ente_sel != 'Todas':
        df = df[df['tipo_ente'] == tipo_ente_sel]
    tipos_fin = ['Todos'] + sorted(df['RGF_clasificacion'].dropna().unique().tolist())
    tipo_fin_sel = st.sidebar.selectbox('Tipo de financiamiento', tipos_fin, index=0)
    # Sliders después
    df['fecha_contratacion'] = pd.to_datetime(df['fecha_contratacion'], errors='coerce')
    df['año'] = df['fecha_contratacion'].dt.year
    min_year = int(df['fecha_contratacion'].dt.year.min())
    max_year = int(df['fecha_contratacion'].dt.year.max())
    year_range = st.sidebar.slider('Filtrar por año de contratación', min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1)
    min_usd_m = float(df['valor_usd'].min()) / 1_000_000
    max_usd_m = float(df['valor_usd'].max()) / 1_000_000
    valor_usd_range_m = st.sidebar.slider('Filtrar por valor USD (millones)', min_value=min_usd_m, max_value=max_usd_m, value=(min_usd_m, max_usd_m), step=0.1, format='%.2f')
    min_tiempo = float(df['tiempo_prestamo'].min())
    max_tiempo = float(df['tiempo_prestamo'].max())
    tiempo_prestamo_range = st.sidebar.slider('Filtrar por tiempo de préstamo (años)', min_value=min_tiempo, max_value=max_tiempo, value=(min_tiempo, max_tiempo), step=1.0, format='%.1f')
    # Aplicar filtros
    df = df[(df['fecha_contratacion'].dt.year >= year_range[0]) & (df['fecha_contratacion'].dt.year <= year_range[1])]
    df = df[(df['valor_usd'] >= valor_usd_range_m[0] * 1_000_000) & (df['valor_usd'] <= valor_usd_range_m[1] * 1_000_000)]
    df = df[(df['tiempo_prestamo'] >= tiempo_prestamo_range[0]) & (df['tiempo_prestamo'] <= tiempo_prestamo_range[1])]
    if tipo_fin_sel != 'Todos':
        df = df[df['RGF_clasificacion'] == tipo_fin_sel]
    # --- GRAFICOS DE DONUTS EN SUBPLOTS ---
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    regiones_unicas = list(df['região'].dropna().unique())
    n = len(regiones_unicas)
    if n == 0:
        st.info('No hay datos para mostrar.')
    else:
        # --- LEYENDA PERSONALIZADA ---
        from collections import OrderedDict
        color_map = {
            'Agricultural development': '#003049',
            'Education policy and administrative management': '#034078',
            'Electric power transmission and distribution': '#2176ff',
            'Energy generation, renewable sources - multiple technologies': '#38b000',
            'Environmental policy and administrative management': '#4B4E6D',
            'Formal sector financial intermediaries': '#c1121f',
            'General infrastructure': '#246a73',
            'Housing policy and administrative management': '#b7bdc1',
            'Medical services': '#fdf0d5',
            'Rural development': '#5E6472',
            'Sanitation - large systems': '#6C584C',
            'Security system management and reform': '#3D405B',
            'Social protection and welfare services policy, planning and administration': '#669bbc',
            'Technological research and development': '#415A77',
            'Tourism policy and administrative management': '#B7BDCB',
            'Transport policy, planning and administration': '#1B263B',
            'Urban development': '#274060',
            'Waste management/disposal': '#2C363F',
            'Water supply - large systems': '#264653',
            'Otros': '#b7bdc1',
        }
        # Recolectar sectores presentes en los datos filtrados
        sectores_presentes = OrderedDict()
        for idx, reg in enumerate(regiones_unicas):
            df_reg = df[df['região'] == reg]
            top_sectores = df_reg.groupby('sector')['valor_usd'].sum().nlargest(5)
            for nombre in top_sectores.index:
                sectores_presentes[nombre] = color_map.get(nombre, '#CCCCCC')
            if df_reg[~df_reg['sector'].isin(top_sectores.index)].shape[0] > 0:
                sectores_presentes['Otros'] = color_map.get('Otros', '#b7bdc1')
        # --- GRAFICOS DE DONUTS Y LEYENDA EN LA SEGUNDA FILA ---
        cols = 3
        n = len(regiones_unicas)
        rows = (n + cols - 1) // cols
        # Renderizar primera fila de donuts
        if n > 0:
            donut_cols = st.columns(cols)
            for i in range(min(cols, n)):
                with donut_cols[i]:
                    reg = regiones_unicas[i]
                    df_reg = df[df['região'] == reg]
                    top_sectores = df_reg.groupby('sector')['valor_usd'].sum().nlargest(5)
                    otros = df_reg[~df_reg['sector'].isin(top_sectores.index)]['valor_usd'].sum()
                    labels = list(top_sectores.index)
                    values = list(top_sectores.values)
                    if otros > 0:
                        labels.append('Otros')
                        values.append(otros)
                    # Limitar a solo 6 labels (top 5 + Otros)
                    labels = labels[:6]
                    values = values[:6]
                    import plotly.express as px
                    default_colors = px.colors.qualitative.Plotly
                    def get_color(nombre):
                        if nombre in color_map:
                            return color_map[nombre]
                        else:
                            all_labels = list(color_map.keys()) + sorted(set(sectores_presentes.keys()) - set(color_map.keys()))
                            idx = all_labels.index(nombre)
                            return default_colors[idx % len(default_colors)]
                    colors = [get_color(label) for label in labels]
                    hovertexts = [f"{n}<br>{v/1_000_000:.2f} M USD" for n, v in zip(list(top_sectores.index), list(top_sectores.values))]
                    if otros > 0:
                        hovertexts.append(f'Otros<br>{otros/1_000_000:.2f} M USD')
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.75,
                            textinfo='percent',
                            textposition='inside',
                            showlegend=False,
                            marker=dict(colors=colors, line=dict(color='black', width=2)),
                            hovertext=hovertexts,
                            hoverinfo='text+percent',
                            domain={'x': [0.08, 0.92], 'y': [0.08, 0.92]}
                        )
                    ])
                    fig.update_traces(textfont_size=12, textinfo='percent')
                    fig.update_layout(
                        margin=dict(t=0, b=0, l=0, r=0),
                        height=260,
                    )
                    theme_base = st.get_option('theme.base')
                    region_title_color = 'white' if theme_base == 'dark' else 'black'
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text=f"<b>{reg}</b>",
                        showarrow=False,
                        font=dict(size=22, color=region_title_color),
                        xref="paper",
                        yref="paper",
                        xanchor="center",
                        yanchor="middle",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        # Renderizar segunda fila: leyenda + donuts restantes
        if n > cols:
            donut_cols2 = st.columns(cols)
            with donut_cols2[0]:
                legend_html = "<div style='display: flex; flex-direction: column; gap: 8px; align-items: flex-start; margin-top: 10px; margin-bottom: 30px;'>"
                def get_color_leyenda(nombre):
                    if nombre in color_map:
                        return color_map[nombre]
                    else:
                        all_labels = list(color_map.keys()) + sorted(set(sectores_presentes.keys()) - set(color_map.keys()))
                        idx = all_labels.index(nombre)
                        return default_colors[idx % len(default_colors)]
                for nombre, _ in sectores_presentes.items():
                    color = get_color_leyenda(nombre)
                    legend_html += f"<div style='display: flex; align-items: center; gap: 6px;'><div style='width: 14px; height: 14px; background: {color}; border-radius: 3px; border: 1px solid #888;'></div><span style='font-size: 13px;'>{nombre}</span></div>"
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)
            donuts_restantes = n - cols
            for j in range(cols, n):
                col_idx = j - cols + 1
                if donuts_restantes == 1 and col_idx == 2:
                    continue
                with donut_cols2[col_idx]:
                    reg = regiones_unicas[j]
                    df_reg = df[df['região'] == reg]
                    top_sectores = df_reg.groupby('sector')['valor_usd'].sum().nlargest(5)
                    otros = df_reg[~df_reg['sector'].isin(top_sectores.index)]['valor_usd'].sum()
                    labels = list(top_sectores.index)
                    values = list(top_sectores.values)
                    if otros > 0:
                        labels.append('Otros')
                        values.append(otros)
                    # Limitar a solo 6 labels (top 5 + Otros)
                    labels = labels[:6]
                    values = values[:6]
                    import plotly.express as px
                    default_colors = px.colors.qualitative.Plotly
                    def get_color(nombre):
                        if nombre in color_map:
                            return color_map[nombre]
                        else:
                            all_labels = list(color_map.keys()) + sorted(set(sectores_presentes.keys()) - set(color_map.keys()))
                            idx = all_labels.index(nombre)
                            return default_colors[idx % len(default_colors)]
                    colors = [get_color(label) for label in labels]
                    hovertexts = [f"{n}<br>{v/1_000_000:.2f} M USD" for n, v in zip(labels, values)]
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.75,
                            textinfo='percent',
                            textposition='inside',
                            showlegend=False,
                            marker=dict(colors=colors, line=dict(color='black', width=2)),
                            hovertext=hovertexts,
                            hoverinfo='text+percent',
                            domain={'x': [0.08, 0.92], 'y': [0.08, 0.92]}
                        )
                    ])
                    fig.update_traces(textfont_size=12, textinfo='percent')
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=260)
                    theme_base = st.get_option('theme.base')
                    region_title_color = 'white' if theme_base == 'dark' else 'black'
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text=f"<b>{reg}</b>",
                        showarrow=False,
                        font=dict(size=22, color=region_title_color),
                        xref="paper",
                        yref="paper",
                        xanchor="center",
                        yanchor="middle",
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif page == 'Financiador por Sector':
    header_cols = st.columns([0.12, 0.88])
    with header_cols[0]:
        st.image(logo_img, width=60)
    with header_cols[1]:
        st.markdown('<h1 style="margin:0; padding:0;">Financiador por Sector</h1>', unsafe_allow_html=True)
    st.write('Análisis de sectores según el financiador (solo FONPLATA, BIRF, CAF, NDB y BID).')
    df = data.copy()
    # --- FILTROS EN SIDEBAR ---
    st.sidebar.markdown('---')
    st.sidebar.subheader('Filtros')
    # Filtro Garantía Soberana
    garantia_opts = ['Todos', 'Si', 'No']
    garantia_sel = st.sidebar.selectbox('Garantía Soberana', garantia_opts, index=0)
    if garantia_sel != 'Todos':
        df = df[df['garantia_soberana'] == garantia_sel]
    # Filtros selectbox primero
    tipo_entes = ['Todas'] + sorted(df['tipo_ente'].dropna().unique().tolist())
    tipo_ente_sel = st.sidebar.selectbox('Filtrar por tipo de ente', tipo_entes, index=0)
    if tipo_ente_sel != 'Todas':
        df = df[df['tipo_ente'] == tipo_ente_sel]
    regiones = ['Todas'] + sorted(df['região'].dropna().unique().tolist())
    region_sel = st.sidebar.selectbox('Filtrar por región', regiones, index=0)
    tipos_fin = ['Todos'] + sorted(df['RGF_clasificacion'].dropna().unique().tolist())
    tipo_fin_sel = st.sidebar.selectbox('Tipo de financiamiento', tipos_fin, index=0)
    # Sliders después
    df['fecha_contratacion'] = pd.to_datetime(df['fecha_contratacion'], errors='coerce')
    df['año'] = df['fecha_contratacion'].dt.year
    min_year = int(df['fecha_contratacion'].dt.year.min())
    max_year = int(df['fecha_contratacion'].dt.year.max())
    year_range = st.sidebar.slider('Filtrar por año de contratación', min_value=min_year, max_value=max_year, value=(min_year, max_year), step=1)
    min_usd_m = float(df['valor_usd'].min()) / 1_000_000
    max_usd_m = float(df['valor_usd'].max()) / 1_000_000
    valor_usd_range_m = st.sidebar.slider('Filtrar por valor USD (millones)', min_value=min_usd_m, max_value=max_usd_m, value=(min_usd_m, max_usd_m), step=0.1, format='%.2f')
    min_tiempo = float(df['tiempo_prestamo'].min())
    max_tiempo = float(df['tiempo_prestamo'].max())
    tiempo_prestamo_range = st.sidebar.slider('Filtrar por tiempo de préstamo (años)', min_value=min_tiempo, max_value=max_tiempo, value=(min_tiempo, max_tiempo), step=1.0, format='%.1f')
    # Aplicar filtros
    df = df[(df['fecha_contratacion'].dt.year >= year_range[0]) & (df['fecha_contratacion'].dt.year <= year_range[1])]
    df = df[(df['valor_usd'] >= valor_usd_range_m[0] * 1_000_000) & (df['valor_usd'] <= valor_usd_range_m[1] * 1_000_000)]
    df = df[(df['tiempo_prestamo'] >= tiempo_prestamo_range[0]) & (df['tiempo_prestamo'] <= tiempo_prestamo_range[1])]
    if region_sel != 'Todas':
        df = df[df['região'] == region_sel]
    if tipo_fin_sel != 'Todos':
        df = df[df['RGF_clasificacion'] == tipo_fin_sel]
    financiadores = ['FONPLATA', 'BIRF', 'CAF', 'NDB', 'BID']
    df = df[df['nombre_acreedor'].isin(financiadores)]
    # --- SOLO FINANCIADORES SELECCIONADOS ---
    # --- GRAFICOS DE DONUTS EN SUBPLOTS ---
    import plotly.graph_objects as go
    from collections import OrderedDict
    color_map = {
        'Agricultural development': '#003049',
        'Education policy and administrative management': '#034078',
        'Electric power transmission and distribution': '#2176ff',
        'Energy generation, renewable sources - multiple technologies': '#38b000',
        'Environmental policy and administrative management': '#4B4E6D',
        'Formal sector financial intermediaries': '#c1121f',
        'General infrastructure': '#246a73',
        'Housing policy and administrative management': '#b7bdc1',
        'Medical services': '#fdf0d5',
        'Rural development': '#5E6472',
        'Sanitation - large systems': '#6C584C',
        'Security system management and reform': '#3D405B',
        'Social protection and welfare services policy, planning and administration': '#669bbc',
        'Technological research and development': '#415A77',
        'Tourism policy and administrative management': '#B7BDCB',
        'Transport policy, planning and administration': '#1B263B',
        'Urban development': '#274060',
        'Waste management/disposal': '#2C363F',
        'Water supply - large systems': '#264653',
        'Otros': '#b7bdc1',
    }
    financiadores_unicos = financiadores
    n = len(financiadores_unicos)
    if n == 0:
        st.info('No hay datos para mostrar.')
    else:
        # --- LEYENDA PERSONALIZADA ---
        sectores_presentes = OrderedDict()
        for idx, fin in enumerate(financiadores_unicos):
            df_fin = df[df['nombre_acreedor'] == fin]
            top_sectores = df_fin.groupby('sector')['valor_usd'].sum().nlargest(5)
            for nombre in top_sectores.index:
                sectores_presentes[nombre] = color_map.get(nombre, '#CCCCCC')
            if df_fin[~df_fin['sector'].isin(top_sectores.index)].shape[0] > 0:
                sectores_presentes['Otros'] = color_map.get('Otros', '#b7bdc1')
        cols = 3
        n = len(financiadores_unicos)
        rows = (n + cols - 1) // cols
        # Renderizar primera fila de donuts
        if n > 0:
            donut_cols = st.columns(cols)
            for i in range(min(cols, n)):
                with donut_cols[i]:
                    fin = financiadores_unicos[i]
                    df_fin = df[df['nombre_acreedor'] == fin]
                    top_sectores = df_fin.groupby('sector')['valor_usd'].sum().nlargest(5)
                    otros = df_fin[~df_fin['sector'].isin(top_sectores.index)]['valor_usd'].sum()
                    labels = list(top_sectores.index)
                    values = list(top_sectores.values)
                    if otros > 0:
                        labels.append('Otros')
                        values.append(otros)
                    # Limitar a solo 6 labels (top 5 + Otros)
                    labels = labels[:6]
                    values = values[:6]
                    import plotly.express as px
                    default_colors = px.colors.qualitative.Plotly
                    def get_color(nombre):
                        if nombre in color_map:
                            return color_map[nombre]
                        else:
                            all_labels = list(color_map.keys()) + sorted(set(sectores_presentes.keys()) - set(color_map.keys()))
                            idx = all_labels.index(nombre)
                            return default_colors[idx % len(default_colors)]
                    colors = [get_color(label) for label in labels]
                    hovertexts = [f"{n}<br>{v/1_000_000:.2f} M USD" for n, v in zip(labels, values)]
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.75,
                            textinfo='percent',
                            textposition='inside',
                            showlegend=False,
                            marker=dict(colors=colors, line=dict(color='black', width=2)),
                            hovertext=hovertexts,
                            hoverinfo='text+percent',
                            domain={'x': [0.08, 0.92], 'y': [0.08, 0.92]}
                        )
                    ])
                    fig.update_traces(textfont_size=12, textinfo='percent')
                    fig.update_layout(
                        margin=dict(t=0, b=0, l=0, r=0),
                        height=260,
                    )
                    theme_base = st.get_option('theme.base')
                    fin_title_color = 'white' if theme_base == 'dark' else 'black'
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text=f"<b>{fin}</b>",
                        showarrow=False,
                        font=dict(size=22, color=fin_title_color),
                        xref="paper",
                        yref="paper",
                        xanchor="center",
                        yanchor="middle",
                    )
                    st.plotly_chart(fig, use_container_width=True)
        # Renderizar segunda fila: leyenda + donuts restantes
        if n > cols:
            donut_cols2 = st.columns(cols)
            with donut_cols2[0]:
                legend_html = "<div style='display: flex; flex-direction: column; gap: 8px; align-items: flex-start; margin-top: 10px; margin-bottom: 30px;'>"
                def get_color_leyenda(nombre):
                    if nombre in color_map:
                        return color_map[nombre]
                    else:
                        all_labels = list(color_map.keys()) + sorted(set(sectores_presentes.keys()) - set(color_map.keys()))
                        idx = all_labels.index(nombre)
                        return default_colors[idx % len(default_colors)]
                for nombre, _ in sectores_presentes.items():
                    color = get_color_leyenda(nombre)
                    legend_html += f"<div style='display: flex; align-items: center; gap: 6px;'><div style='width: 14px; height: 14px; background: {color}; border-radius: 3px; border: 1px solid #888;'></div><span style='font-size: 13px;'>{nombre}</span></div>"
                legend_html += "</div>"
                st.markdown(legend_html, unsafe_allow_html=True)
            donuts_restantes = n - cols
            for j in range(cols, n):
                col_idx = j - cols
                if col_idx >= cols:
                    break  # Nunca excedas el número de columnas
                with donut_cols2[col_idx]:
                    fin = financiadores_unicos[j]
                    df_fin = df[df['nombre_acreedor'] == fin]
                    top_sectores = df_fin.groupby('sector')['valor_usd'].sum().nlargest(5)
                    otros = df_fin[~df_fin['sector'].isin(top_sectores.index)]['valor_usd'].sum()
                    labels = list(top_sectores.index)
                    values = list(top_sectores.values)
                    if otros > 0:
                        labels.append('Otros')
                        values.append(otros)
                    labels = labels[:6]
                    values = values[:6]
                    import plotly.express as px
                    default_colors = px.colors.qualitative.Plotly
                    def get_color(nombre):
                        if nombre in color_map:
                            return color_map[nombre]
                        else:
                            all_labels = list(color_map.keys()) + sorted(set(sectores_presentes.keys()) - set(color_map.keys()))
                            idx = all_labels.index(nombre)
                            return default_colors[idx % len(default_colors)]
                    colors = [get_color(label) for label in labels]
                    hovertexts = [f"{n}<br>{v/1_000_000:.2f} M USD" for n, v in zip(labels, values)]
                    fig = go.Figure(data=[
                        go.Pie(
                            labels=labels,
                            values=values,
                            hole=0.75,
                            textinfo='percent',
                            textposition='inside',
                            showlegend=False,
                            marker=dict(colors=colors, line=dict(color='black', width=2)),
                            hovertext=hovertexts,
                            hoverinfo='text+percent',
                            domain={'x': [0.08, 0.92], 'y': [0.08, 0.92]}
                        )
                    ])
                    fig.update_traces(textfont_size=12, textinfo='percent')
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=260)
                    theme_base = st.get_option('theme.base')
                    fin_title_color = 'white' if theme_base == 'dark' else 'black'
                    fig.add_annotation(
                        x=0.5,
                        y=0.5,
                        text=f"<b>{fin}</b>",
                        showarrow=False,
                        font=dict(size=22, color=fin_title_color),
                        xref="paper",
                        yref="paper",
                        xanchor="center",
                        yanchor="middle",
                    )
                    st.plotly_chart(fig, use_container_width=True)

elif page == 'Dispersion':
    header_cols = st.columns([0.12, 0.88])
    with header_cols[0]:
        st.image(logo_img, width=60)
    with header_cols[1]:
        st.markdown('<h1 style="margin:0; padding:0;">Dispersión: Valor USD vs Tiempo de Préstamo</h1>', unsafe_allow_html=True)
    st.write('Cada punto representa un financiamiento. Eje X: Valor USD, Eje Y: Tiempo de préstamo. Color: Financiador.')
    df = data.copy()
    # --- FILTROS EN SIDEBAR ---
    st.sidebar.markdown('---')
    st.sidebar.subheader('Filtros')
    # Filtro Garantía Soberana
    garantia_opts = ['Todos', 'Si', 'No']
    garantia_sel = st.sidebar.selectbox('Garantía Soberana', garantia_opts, index=0)
    if garantia_sel != 'Todos':
        df = df[df['garantia_soberana'] == garantia_sel]
    # Filtros selectbox primero
    tipo_entes = ['Todas'] + sorted(df['tipo_ente'].dropna().unique().tolist())
    tipo_ente_sel = st.sidebar.selectbox('Filtrar por tipo de ente', tipo_entes, index=0)
    if tipo_ente_sel != 'Todas':
        df = df[df['tipo_ente'] == tipo_ente_sel]
    # Sliders después
    min_usd_m = float(df['valor_usd'].min()) / 1_000_000
    max_usd_m = float(df['valor_usd'].max()) / 1_000_000
    valor_usd_range_m = st.sidebar.slider('Filtrar por valor USD (millones)', min_value=min_usd_m, max_value=max_usd_m, value=(min_usd_m, max_usd_m), step=0.1, format='%.2f')
    min_tiempo = float(df['tiempo_prestamo'].min())
    max_tiempo = float(df['tiempo_prestamo'].max())
    tiempo_prestamo_range = st.sidebar.slider('Filtrar por tiempo de préstamo (años)', min_value=min_tiempo, max_value=max_tiempo, value=(min_tiempo, max_tiempo), step=1.0, format='%.1f')
    # Aplicar filtros
    df = df[(df['valor_usd'] >= valor_usd_range_m[0] * 1_000_000) & (df['valor_usd'] <= valor_usd_range_m[1] * 1_000_000)]
    df = df[(df['tiempo_prestamo'] >= tiempo_prestamo_range[0]) & (df['tiempo_prestamo'] <= tiempo_prestamo_range[1])]
    financiadores_disp = ['FONPLATA', 'Caixa', 'BNDS', 'BID', 'BIRF', 'CAF', 'NDB', 'Banco do Brasil S/A']
    df = df[df['nombre_acreedor'].isin(financiadores_disp)]
    # Convertir valor_usd a millones para el scatter
    df['valor_usd_millones'] = df['valor_usd'] / 1_000_000
    color_map = {
        'BIRF': '#003049',
        'BID': '#034078',
        'FONPLATA': '#c1121f',
        'CAF': '#38b000',
        'Caixa': '#ffc600',
        'Unión': '#f18701',
        'BNDS': '#246a73',
        'Otros': '#b7bdc1',
        'Banco do Brasil S/A': '#2176ff',
        'Bank of America Merrill Lynch': '#fdf0d5',
        'NDB': '#6a4c93',
    }
    import plotly.express as px
    fig = px.scatter(
        df,
        x='tiempo_prestamo',
        y='valor_usd_millones',
        color='nombre_acreedor',
        color_discrete_map=color_map,
        hover_data=['sector', 'região', 'fecha_contratacion'],
        labels={'valor_usd_millones': 'Valor USD (millones)', 'tiempo_prestamo': 'Tiempo de Préstamo (años)', 'nombre_acreedor': 'Financiador'},
        title='Dispersión de Financiamientos por Financiador',
        height=700,
        size_max=10,
        size=[6]*len(df)  # Tamaño fijo aún más pequeño para todos los puntos
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='black')))
    fig.update_layout(title_x=0.5, plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True)

elif page == 'Box Plots':
    header_cols = st.columns([0.12, 0.88])
    with header_cols[0]:
        st.image(logo_img, width=60)
    with header_cols[1]:
        st.markdown('<h1 style="margin:0; padding:0;">Box Plots: Comparación de Financiadores</h1>', unsafe_allow_html=True)
    st.write('Comparación de la distribución del valor USD entre los principales financiadores.')
    df = data.copy()
    # --- FILTROS EN SIDEBAR ---
    st.sidebar.markdown('---')
    st.sidebar.subheader('Filtros')
    # Filtro Garantía Soberana
    garantia_opts = ['Todos', 'Si', 'No']
    garantia_sel = st.sidebar.selectbox('Garantía Soberana', garantia_opts, index=0)
    if garantia_sel != 'Todos':
        df = df[df['garantia_soberana'] == garantia_sel]
    # Filtros selectbox primero
    tipo_entes = ['Todas'] + sorted(df['tipo_ente'].dropna().unique().tolist())
    tipo_ente_sel = st.sidebar.selectbox('Filtrar por tipo de ente', tipo_entes, index=0)
    if tipo_ente_sel != 'Todas':
        df = df[df['tipo_ente'] == tipo_ente_sel]
    # Sliders después
    financiadores_box = ['Caixa', 'BNDS', 'FONPLATA', 'BID', 'CAF', 'BIRF', 'NDB']
    df = df[df['nombre_acreedor'].isin(financiadores_box)]
    min_tiempo = float(df['tiempo_prestamo'].min())
    max_tiempo = float(df['tiempo_prestamo'].max())
    tiempo_prestamo_range = st.sidebar.slider('Filtrar por tiempo de préstamo (años)', min_value=min_tiempo, max_value=max_tiempo, value=(min_tiempo, max_tiempo), step=1.0, format='%.1f')
    df = df[(df['tiempo_prestamo'] >= tiempo_prestamo_range[0]) & (df['tiempo_prestamo'] <= tiempo_prestamo_range[1])]
    # Convertir valor_usd a millones para el boxplot
    df['valor_usd_millones'] = df['valor_usd'] / 1_000_000
    color_map = {
        'BIRF': '#003049',
        'BID': '#034078',
        'FONPLATA': '#c1121f',
        'CAF': '#38b000',
        'Caixa': '#ffc600',
        'BNDS': '#246a73',
        'NDB': '#6a4c93',
    }
    import plotly.express as px
    fig = px.box(
        df,
        x='nombre_acreedor',
        y='valor_usd_millones',
        color='nombre_acreedor',
        color_discrete_map=color_map,
        points='all',
        labels={'nombre_acreedor': 'Financiador', 'valor_usd_millones': 'Valor USD (millones)'},
        title='Distribución del Valor USD por Financiador',
        height=700
    )
    fig.update_traces(marker=dict(size=6, opacity=0.5, line=dict(width=1, color='black')))
    fig.update_layout(title_x=0.5, plot_bgcolor='white')
    st.plotly_chart(fig, use_container_width=True) 
