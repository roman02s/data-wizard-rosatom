import os
import uuid

import torch

import numpy as np
import pandas as pd

import streamlit as st
import networkx as nx

import plotly.express as px
from src.visualizazation import create_graph_viz
from src.create_graph import create_graph, add_answer, graph_to_dataframe
# from src.generate_embegging import get_embedding, get_answers_clustering
# from src.answer_clustering import AnswersClustering
from src.toxik import detox_text
from clusterisation import clusterisation


import pandas as pd
import torch.nn.functional as F
import torch
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


# from transformers import GPT2Tokenizer, T5ForConditionalGeneration 

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
# import time
# start = time.time()
# detox_text("")
# print("SUCCESS IMPOT DETOX", time.time() - start)
# start = time.time()
# tokenizer_for_summary = GPT2Tokenizer.from_pretrained('ai-forever/FRED-T5-1.7B',eos_token='</s>')
# print("SUCCESS IMPORT TOKENIZER SUMMARY", time.time() - start)
# start = time.time()
# model_for_summary = T5ForConditionalGeneration.from_pretrained('ai-forever/FRED-T5-1.7B')
# print("SUCCESS IMPORT MODEL SUMMARY", time.time() - start)

# save the models to disk
# Сохранение токенизатора и модели на диск
# tokenizer_for_summary.save_pretrained('tokenizer_for_summary')  # Замените 'path_to_save_tokenizer' на путь к папке, где вы хотите сохранить токенизатор
# model_for_summary.save_pretrained('model_for_summary.pth')  # Замените 'path_to_save_model' на путь к папке, где вы хотите сохранить модель


# Пример загрузки сохраненной модели и токенизатора позже
# tokenizer_for_summary = GPT2Tokenizer.from_pretrained('tokenizer_for_summary')
# model_for_summary = T5ForConditionalGeneration.from_pretrained('model_for_summary.pth')


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
# if "tokenizer_for_summary" not in st.session_state:
#     st.session_state.tokenizer_for_summary = GPT2Tokenizer.from_pretrained('tokenizer_for_summary')
# if "model_for_summary" not in st.session_state:
    # st.session_state.model_for_summary = T5ForConditionalGeneration.from_pretrained('model_for_summary.pth')
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large')
if "model" not in st.session_state:
        st.session_state.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large')

if "G" not in st.session_state:
    st.session_state.G = None


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


import faiss
import math


def get_query_embedding(query_text): # query_text = "query: Какие основания для получения 33 услуги?"
    query_batch_dict = st.session_state.tokenizer([query_text], max_length=512, padding=True, truncation=True, return_tensors='pt')
    query_outputs = st.session_state.model(**query_batch_dict.to("cpu"))
    query_embedding = average_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
    query_embedding = F.normalize(query_embedding, p=2, dim=1)
    return query_embedding.cpu().detach().numpy()

# Создадим поиск с помощью FAISS
def sem_search_faiss(query_text, index, top_k=10):
    query = get_query_embedding(query_text)
    D, I = index.search(query, top_k)
    # resp = np.array(values)[I]
    return D, I
# 
# grouped_data = data.groupby(['Группы', 'Топики'], as_index=False)['Значения'].sum()


# fig = px.bar(grouped_data, x='Группы', y='Значения', color='Топики', title='Тональность обращений')
# fig.update_layout(xaxis_title='Группы', yaxis_title='Значения')
# fig.show()
def barr_plot(data):
    if data.get("sentiment") is None:
        return
    st.markdown("""<center> <h2>Распределение ответов по кластерам</h2></center>""", unsafe_allow_html=True)
    st.write()
    # print(data.groupby(["cluster", "sentiment"], as_index=False)["cluster"].sum())
    grouped_data = data.groupby(["cluster", "sentiment"], as_index=False).count()
    # print(data.groupby(["cluster", "sentiment"], as_index=False))

    # Создаем столбчатую диаграмму
    fig = px.bar(grouped_data, x='cluster', y="answer", color='cluster')

    # Настройка осей и заголовка
    fig.update_layout(xaxis_title='Кластеры', yaxis_title='RКол-во ответов')

    # Отображение диаграммы
    st.plotly_chart(fig)
    

def tonal_plot(data):
    if data.get("sentiment") is None:
        return
    print(data)
    st.markdown("""<center> <h2>Тональность ответов</h2></center>""", unsafe_allow_html=True)
    st.write()
    # print(data.groupby(["cluster", "sentiment"], as_index=False)["cluster"].sum())
    grouped_data = data.groupby(['cluster', 'sentiment']).size().reset_index(name='count')

    # Создаем столбчатую диаграмму
    fig = px.bar(grouped_data, x='cluster', y='count', color='sentiment',
             labels={'count': 'Количество', 'cluster': 'Кластер', 'sentiment': 'Настроение'},
             title='Количество записей по кластерам и настроениям')

    # Настройка осей и заголовка
    fig.update_layout(xaxis_title='Кластеры', yaxis_title='Тональность ответов')
    
    # Отображение диаграммы
    st.plotly_chart(fig)
    

def py_chart(data):
    if data.get("sentiment") is None:
        return
    elif data.get("cluster") is None:
        return
    st.markdown("""<center> <h2>Распределение ответов по кластерам</h2></center>""", unsafe_allow_html=True)
    import plotly.graph_objects as go
    
    fig =go.Figure(go.Sunburst(
        labels=data["cluster"],
        parents=data["question"],
        values=[50 for i in data["answer"]],
        branchvalues="total",
    ))
    # fig.update_layout(margin = dict(t=0, l=0, r=0, b=0))
    
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
    top5 = st.button("Показать топ 5 вопросов")
    if top5:
        result_emb = torch.load("../data/embds.pt")
        df = pd.read_csv("../data/sim_data.csv")
        dim = result_emb.shape[1] #передаем размерность пр-ва
        size = result_emb.shape[0] #размер индекса

        index = faiss.IndexFlatL2(dim)
        print(index.ntotal)  # пока индекс пустой
        index.add(result_emb.cpu().detach().numpy())
        print(index.ntotal)  # теперь в нем sentence_embeddings.shape[0] векторов
        
        result = sem_search_faiss(
                        query_text=f"query: {client_question_final}",
                        index=index,
                        #  values = np.array(parags),
                        top_k=15
                        )
        df['text'] = df['text'].apply(lambda s: s[8:])
        st.dataframe(df.iloc[result[1][0]]['text'].unique()[:5])
else:
    client_question_final = st.text_input("Введите интересующий вопрос")

try:
    if client_question_final:
        data = labeled[labeled["question"] == client_question_final]
        if len(data) != 0:
            st.session_state.G = None
            G = create_graph(data["question"].values[0], data["answer"].values, data["cluster"].values)
            fig = create_graph_viz(G)
            st.plotly_chart(fig)
            barr_plot(data)
            tonal_plot(data)
            # py_chart(data)
            
        data = questions[questions["question"] == client_question_final]
        if len(data) != 0:
            data["cluster"] = clusterisation(data["answer"])
            st.session_state.G = None
            G = create_graph(data["question"].values[0], data["answer"].values, data["cluster"].values)
            fig = create_graph_viz(G)
            st.plotly_chart(fig)
            barr_plot(data)
            tonal_plot(data)
            # py_chart(data)
        
        if client_question_final and client_question == "Добавить новый вопрос":
            client_answer = st.text_input("Введите ответ")
            check = st.button("Добавить ответ")
            if check and client_answer:
                import time
                a = time.time()
                client_answer = detox_text(client_answer)
                st.write("Обработанный ответ:", client_answer)
                # if len(client_answer) > 33 and st.button("Уменьшить текст"):  
                #     import time
                #     b = time.time()
                #     lm_text= client_answer
                #     input_ids=torch.tensor([st.session_state.tokenizer_for_summary.encode(lm_text)]).to("cpu")
                #     outputs=st.session_state.model_for_summary.generate(input_ids,eos_token_id=st.session_state.tokenizer_for_summary.eos_token_id,early_stopping=True)
                #     client_answer = st.session_state.tokenizer_for_summary.decode(outputs[0][1:])
                #     st.write("Обработанный ответ после суммаризации:", client_answer)
                #     print("Время:", time.time() - b)
                if st.session_state.G is None:
                    client_cluster = clusterisation([client_answer])[-1]
                    # aa_milne_arr = ['neutrals', 'negatives', 'positives']
                    # choise = np.random.choice(aa_milne_arr, 1, p=[0.85, 0.05, 0.1])[0]
                    st.session_state.G = create_graph(client_question_final, [client_answer], [client_cluster])
                else:
                    add_answer(st.session_state.G, client_question_final, clusterisation([client_answer])[-1], client_answer)
                fig = create_graph_viz(st.session_state.G)
                st.plotly_chart(fig)
                data = graph_to_dataframe(st.session_state.G)
                # print(data)
                # print(data.columns)
                barr_plot(data)
                tonal_plot(data)
                # py_chart(data)
            
            
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

