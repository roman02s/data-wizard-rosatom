import os
import uuid

import numpy as np
import pandas as pd

import streamlit as st
import networkx as nx


from src.visualizazation import create_graph_viz
from src.create_graph import create_graph
# from src.generate_embegging import get_embedding, get_answers_clustering

# Создаем граф NetworkX
# data_G = {
#     "Ты выспался?": {
#         "Положительный ответ": ["Да", "Да", "Конечно"],
#         "Нейтральный ответ": ["ок", "Утро...", "Возможно"],
#         "Отрицательный ответ": ["Не спал", "Нет", "Нет"],
#     }
# }

# data_G_colors = {}

# for key, value in data_G.items():
#     data_G_colors[key] = 'rgb(0, 0, 255)'
#     for key_, value_ in value.items():
#         color_vector = np.random.randint(256, size=3)
#         data_G_colors[key_] = f'rgb({color_vector[0]}, {color_vector[1]}, {color_vector[2]})'
#         # data_G_colors[value_] = f'rgb({color_vector[0]}, {color_vector[1]}, {color_vector[2]})'

# print(data_G_colors)

# def create_graph(data, graph, parent=None, depth=0):
#     for key, value in data.items():
#         key_id = uuid.uuid4()
#         graph.add_node(key_id, level=depth, name_1=key, color=data_G_colors[key])
#         if parent is not None:
#             graph.add_edge(parent, key_id)
#         if isinstance(value, dict):
#             create_graph(value, graph, parent=key_id, depth=depth + 1)
#         elif isinstance(value, list):
#             for item in value:
#                 item_id = uuid.uuid4()
#                 print(item, key, data_G_colors[key])
#                 graph.add_node(item_id, level=depth + 1, name_1=item, color=data_G_colors[key])
#                 graph.add_edge(key_id, item_id)


# # Создаем граф
# G = nx.DiGraph()
# create_graph(data_G, G)

# Создаем Dash-приложение




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

# question = st.text_input("Введите интересующий вопрос")

# if st.button("Обновить граф"):
#     # Создаем обратную связь для обновления графа по нажатию кнопки
#     fig = create_graph_viz(G)
#     st.plotly_chart(fig)
# Запускаем Dash-приложение на другом порту

questions = pd.read_csv("../data/all.csv")

if "select_placeholder1" not in st.session_state:
    st.session_state.select_placeholder1 = ""
if "select_placeholder2" not in st.session_state:
    st.session_state.select_placeholder2 = ""
if "G" not in st.session_state:
    st.session_state.G = None

def func_add_node(G, cluster, client_answer):
    k_answer = uuid.uuid4()
    G.add_node(k_answer, level=2, name_1=client_answer, color=G.nodes[cluster]['color'])
    G.add_edge(cluster, k_answer)


# def func_add_node(G, client_question_final, cluster, client_answer):
#     k_cluster = uuid.uuid4()
#     k_answer = uuid.uuid4()
#     G.add_node(k_cluster, level=1, name_1=cluster, color='rgb(0, 0, 255)')
#     G.add_node(k_answer, level=2, name_1=client_answer, color='rgb(0, 0, 255)')
#     G.add_edge(k_cluster, k_answer)
#     # поиск по полю name_1
#     for node in G.nodes():
#         if G.nodes[node]['name_1'] == client_question_final:
#             G.add_edge(node, k_cluster)


QUESTIONS = questions["question"].unique()
client_question = st.radio("Выбрать вопрос", [
    "Добавить новый вопрос",
    "Выбрать из уже существующих вопросов",
])
if client_question == "Выбрать из уже существующих вопросов":
    st.write("Существующие вопросы:")
    client_question_final = st.selectbox(
        label=st.session_state.select_placeholder1,
        key="question", options=QUESTIONS,
    )
else:
    client_question_final = st.text_input("Введите интересующий вопрос")

if client_question_final:
    client_answer = st.text_input("Введите ответ")
    check = st.button("Добавить ответ")
    if check and client_answer:
        st.write("Вы добавили ответ:", client_answer)
        # func_add_node(st.session_state.G, client_question_final, client_answer)
    # fig = create_graph_viz(st.session_state.G)
    # st.plotly_chart(fig)
    
# print(get_embedding("Ты выспался?"))

df = pd.read_csv("../data/result_df.csv")
print(df.head())
print(df.columns)
# Index(['question', 'answer', 'cluster', 'summariz', 'coord_x', 'coord_y',
#        'tonality', 'size', 'color', 'uuid'],
#       dtype='object')
# print(df["cluster"].unique())
# print(df["question"].unique())
# print(df["answer"].unique())
for question in df["question"].unique():
    print(question)
    answers = df[df["question"] == question]["answer"]
    clusters = df[df["question"] == question]["cluster"]
    st.session_state.G = create_graph(question, answers, clusters)
    fig = create_graph_viz(st.session_state.G)
    st.plotly_chart(fig)
    