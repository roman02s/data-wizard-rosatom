import streamlit as st
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import networkx as nx
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# Создаем граф NetworkX
data_G = {
    "Ты выспался?": {
        "Положительный ответ": ["Да", "Да", "Конечно"],
        "Нейтральный ответ": ["ок", "Утро...", "Возможно"],
        "Отрицательный ответ": ["Не спал", "Нет", "Нет"],
    }
}

st.title("Мой голос")


def create_graph(data, graph, parent=None, depth=0):
    for key, value in data.items():
        graph.add_node(key, level=depth, name_1=key)
        if parent is not None:
            graph.add_edge(parent, key)
        if isinstance(value, dict):
            create_graph(value, graph, parent=key, depth=depth + 1)
        elif isinstance(value, list):
            for item in value:
                graph.add_node(item, level=depth + 1, name_1=item)
                graph.add_edge(key, item)


# Создаем граф
G = nx.DiGraph()
create_graph(data_G, G)

# Создаем Dash-приложение


def get_node_size(level):
    if level == 0:
        return 50
    elif level == 1:
        return 35
    else:
        return 15


# Определяем макет Dash-приложения
# app_dash.layout = html.Div([
#     html.H1("Интерактивная визуализация графа"),
#     dcc.Graph(id='graph-visualization',
#               style={'width': '1500px', 'height': '900px', 'margin': '0 auto'}),
#     html.Button('Обновить граф', id='update-button')
# ])

# st.set_page_config(
#     page_title="My App",
#     page_icon=":rocket:",
#     layout="wide",
#     initial_sidebar_state="auto",
#     # bg_color="white"  # Установите желаемый цвет фона (в данном случае, белый)
# )
st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Определяем функцию для создания графической визуализации
def create_graph_viz():
    # Получаем позиции узлов для отображения
    pos = nx.spring_layout(G)

    # Создаем список узлов и ребер для отрисовки
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",  # Обратите внимание на режим markers+text
        hoverinfo="text",
        # marker=dict(
        #     showscale=True,
        #     colorscale='YlGnBu',
        #     size=10,
        #     # colorbar=dict(
        #     #     thickness=15,
        #     #     title='Связи узла',
        #     #     xanchor='left',
        #     #     titleside='right'
        #     # ),
        #     line=dict(width=2)
        # ),
        textfont=dict(size=10, color="black"),  # Настройка стиля текста
    )

    node_trace["marker"]["size"] = []

    edge_trace = go.Scatter(
        x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines"
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace["x"] += (x,)
        node_trace["y"] += (y,)
        # node_info = f'Узел {node}<br>Связи: {len(G.edges(node))}'
        # node_info = f'{node}'
        node_trace["text"] += (G.nodes[node]["name_1"],)
        node_trace["marker"]["size"] += (get_node_size(G.nodes[node]["level"]),)

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace["x"] += (x0, x1, None)
        edge_trace["y"] += (y0, y1, None)

    # Создаем подзаголовок для визуализации
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(node_trace)
    fig.add_trace(edge_trace)

    # Настраиваем внешний вид визуализации
    fig.update_layout(
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)",  # Устанавливаем прозрачный фон
    )

    return fig


# question = st.text_input("Введите интересующий вопрос")

if st.button("Обновить граф"):
    # Создаем обратную связь для обновления графа по нажатию кнопки
    fig = create_graph_viz()
    st.plotly_chart(fig)
# Запускаем Dash-приложение на другом порту

questions = pd.read_csv("../data/all.csv")

if "select_placeholder1" not in st.session_state:
    st.session_state.select_placeholder1 = ""


QUESTIONS = questions["question"].unique()
client_question_new = st.text_input("Введите интересующий вопрос")
with st.expander("Выбрать из уже существующих вопросов"):
    st.write("Существующие вопросы:")
    client_question_exist = st.selectbox(
        label=st.session_state.select_placeholder1,
        key="question", options=QUESTIONS,
    )
if st.button("Выбрать вопрос") and (client_question_new or client_question_exist):
    question = client_question_new if client_question_new else client_question_exist
    st.write("question", question)
    
