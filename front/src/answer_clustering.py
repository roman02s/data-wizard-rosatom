import pickle

import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AffinityPropagation

from generate_embegging import sbert_get_embeddings

NEED_RECALCULATE = [
    1, 2, 3, 4, 5, 7, 10, 15, 20, 25, 40, 50, 60, 80, 100, 125, 150, 200, 250, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 200000
]


class AnswersClustering:
    def __init__(self, params: dict):
        self.id = params["id"]
        self.question = params["question"]
        # для определения тональности
        self.model_tonality = pickle.load("model_tonality.pkl")
        self.vect_tonality = pickle.load("vectorizer_tonality.pkl")
        self.params = params
        # текстовый фиче экстрактор
        self.model = sbert_get_embeddings,
        # тут должен быть конструктор класса
        self.clustering_model = AffinityPropagation(**{"random_state" : 1909,"affinity" : "precomputed",})
        self.emb_list = []
        self.answer_list = []
        self.cluster_list = []
        self.tonalities = []
        self.calculate = 0

    def add_answer(self, answer: str):
        # тут хз, когда будем пересчитывать.
        # можно поставить условие на len(self.answer_list)
        # и пересчитывать в зависимости от этого
        self.answer_list.append(answer)
        self.emb_list.append(self.model([answer]).flatten())

        text_vect = self.vect_tonality.transform([answer])
        self.tonalities.append(self.model_tonality.predict(text_vect)[0])

        self.calculate += 1
        if self.calculate in NEED_RECALCULATE:
        # if recompute:
            self.clustering()
        else:
            self.cluster_list.append(self.clustering_model.predict(self.emb_list[-1].reshape(1, -1))[0])

    def clustering(self):
        self.clustering_model.fit(self.emb_list)
        self.cluster_list = list(self.clustering_model.labels_)

    def get_clusters(self):
        return self.cluster_list

    def get_answers(self):
        return self.answer_list
    
    def get_tonalities(self):
        return self.tonalities
    
    def close():
        pass


