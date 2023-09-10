import os
import uuid

import numpy as np
import pandas as pd

import streamlit as st
import networkx as nx

import plotly.express as px
from src.visualizazation import create_graph_viz
from src.create_graph import create_graph, add_answer, graph_to_dataframe
# from src.generate_embegging import get_embedding, get_answers_clustering
from src.answer_clustering import AnswersClustering

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

st.title("Мой голос")

questions = pd.read_csv("../data/all.csv")
labeled = pd.read_csv("../data/labeled.csv")

if "select_placeholder1" not in st.session_state:
    st.session_state.select_placeholder1 = ""
if "select_placeholder2" not in st.session_state:
    st.session_state.select_placeholder2 = []
if "G" not in st.session_state:
    st.session_state.G = None

def barr_plot(data):
    st.markdown("""<center> <h2>Тональность ответов</h2></center>""", unsafe_allow_html=True)
    st.write()
    grouped_data = data.groupby('sentiment').size().reset_index()

    # Создаем столбчатую диаграмму
    fig = px.bar(grouped_data, x='sentiment', y=0, color='sentiment')

    # Настройка осей и заголовка
    fig.update_layout(xaxis_title='Группы', yaxis_title='Значения')

    # Отображение диаграммы
    st.plotly_chart(fig)

QUESTIONS = np.union1d(
    questions["question"].unique(),
    labeled["question"].unique(),
)
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

try:
    if client_question_final:
        data = labeled[labeled["question"] == client_question_final]
        if len(data) != 0:
            st.session_state.G = None
            G = create_graph(data["question"].values[0], data["answer"].values, data["sentiment"].values)
            fig = create_graph_viz(G)
            st.plotly_chart(fig)
            barr_plot(data)
            
        data = questions[questions["question"] == client_question_final]
        if len(data) != 0:
            st.session_state.G = None
            G = create_graph(data["question"].values[0], data["answer"].values, data["sentiment"].values)
            fig = create_graph_viz(G)
            st.plotly_chart(fig)
            barr_plot(data)
        
        if client_question_final and client_question == "Добавить новый вопрос":
            client_answer = st.text_input("Введите ответ")
            check = st.button("Добавить ответ")
            if check and client_answer:
                st.write("Вы добавили ответ:", client_answer)
                if st.session_state.G is None:
                    aa_milne_arr = ['neutrals', 'negatives', 'positives']
                    choise = np.random.choice(aa_milne_arr, 1, p=[0.85, 0.05, 0.1])[0]
                    st.session_state.G = create_graph(client_question_final, [client_answer], [choise])
                else:
                    add_answer(st.session_state.G, client_question_final, "neutrals", client_answer)
                fig = create_graph_viz(st.session_state.G)
                st.plotly_chart(fig)
                data = graph_to_dataframe(st.session_state.G)
                print(data)
                print(data.columns)
                barr_plot(data)
            
            
            # positives, negatives, neutrals
except BaseException as err:
    st.write("Вопрос не найден")
    st.write(err)
    raise err

# if client_question_final:
#     client_answer = st.text_input("Введите ответ")
#     check = st.button("Добавить ответ")
#     if check and client_answer:
#         st.write("Вы добавили ответ:", client_answer)
#         id_quest = questions[questions["question"] == client_question_final]["id"].values[0]
#         print("id_quest", id_quest)
#         # params = {"id": id_quest,
#         #     "question": client_question_final,
#         #     "model": get_embedding,
#         #     "clustering": get_answers_clustering
#         # }
#         clusters = pd.read_csv(f"../data/clusters/{id_quest}.clusters.csv")
#         cluster = clusters[clusters["answer"] == client_answer]["cluster"]
#         if len(cluster) == 0:
#             # топ кластер
#             cluster = clusters["cluster"].value_counts().index[0]
#         print("cluster", cluster)
#         if st.session_state.G is None:
#             st.session_state.G = nx.DiGraph()
#             st.session_state.G.add_node(uuid.uuid4(), level=0, name_1=client_question_final, color='rgb(255, 0, 0)')
#         print("Before add_answer", st.session_state.G.nodes, st.session_state.G.edges,
#               cluster, client_answer)
#         add_answer(st.session_state.G, client_question_final, cluster, client_answer)
#         fig = create_graph_viz(st.session_state.G)
#         st.plotly_chart(fig)
    

# df = pd.read_csv("../data/result_df.csv")
# print(df.head())
# print(df.columns)
# for question in df["question"].unique():
#     print(question)
#     answers = df[df["question"] == question]["answer"]
#     clusters = df[df["question"] == question]["cluster"]
#     st.session_state.G = create_graph(question, answers, clusters)
#     fig = create_graph_viz(st.session_state.G)
#     st.plotly_chart(fig)

