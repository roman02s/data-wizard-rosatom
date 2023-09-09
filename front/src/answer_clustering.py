import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

NEED_RECALCULATE = [
    2, 3, 4, 5, 7, 10, 15, 20, 25, 40, 50, 60, 80, 100, 125, 150, 200, 250, 400, 600, 800, 1000, 1500, 2000, 3000, 5000, 10000, 20000, 50000, 100000, 200000
]

class AnswersClustering:
    def __init__(self, params: dict):
        self.params = params
        # текстовый фиче экстрактор
        self.model = params["model"]
        # тут должен быть конструктор класса
        self.clustering_model = self.params["clustering"](**params["clustering_params"])
        self.emb_list = []
        self.answer_list = []
        self.distance_matrix = None
        self.cluster_list = []
        self.calculate = 0

    def add_answer(self, answer: str, recompute: bool):
        # тут хз, когда будем пересчитывать.
        # можно поставить условие на len(self.answer_list)
        # и пересчитывать в зависимости от этого
        self.answer_list.append(answer)
        self.emb_list.append(self.model([answer]).flatten())
        
        self.calculate += 1
        if recompute and self.calculate in NEED_RECALCULATE:
            self.clustering()
        else:
            self.cluster_list.append(self.clustering_model.predict(self.emb_list[-1]))

    def clustering(self):
        self.clustering_model.fit(self.distance_matrix)
        self.cluster_list = list(self.clustering_model.labels_)

    def get_clusters(self):
        return self.cluster_list

    def get_answers(self):
        return self.answer_list

