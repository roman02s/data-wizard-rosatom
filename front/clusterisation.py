import torch

import numpy as np
import pandas as pd

from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_distances
from transformers import AutoTokenizer, AutoModel



device = ("cuda" if torch.cuda.is_available() else "cpu")
sbert_tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/sbert_large_nlu_ru")
sbert_model = AutoModel.from_pretrained("sberbank-ai/sbert_large_nlu_ru").to(device)

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


def sbert_get_embeddings(texts):
    tokens = sbert_tokenizer(list(texts), padding=True, truncation=True, max_length=24, return_tensors='pt').to("cpu")
    with torch.no_grad():
        meta = sbert_model(**tokens)
    embs = np.asarray(mean_pooling(meta, tokens['attention_mask']).to("cpu"))
    return embs


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

    def add_answer(self, answer: str, recompute: bool):
        # тут хз, когда будем пересчитывать.
        # можно поставить условие на len(self.answer_list)
        # и пересчитывать в зависимости от этого
        self.answer_list.append(answer)
        self.emb_list.append(self.model([answer]).flatten())

        if recompute:
            self.compute_distances()
            self.clustering()
        else:
            self.cluster_list.append(self.clustering_model.predict(self.emb_list[-1]))

    def compute_distances(self):
        self.distance_matrix = cosine_distances(self.emb_list)

    def clustering(self):
        self.clustering_model.fit(self.distance_matrix)
        self.cluster_list = list(self.clustering_model.labels_)

    def get_clusters(self):
        return self.cluster_list

    def get_answers(self):
        return self.answer_list


params = {
    "model" : sbert_get_embeddings,
    "clustering" : AffinityPropagation,
    "clustering_params" : {
        "random_state" : 1909,
        "affinity" : "precomputed",
        "damping": 0.8,
    }
}
df = pd.read_csv("../data/all.csv")



def clusterisation(answers: "List[str]") -> "List[str]":
    test_ = AnswersClustering(params)
    for answ in answers:
        test_.add_answer(answ, True)
    clusters = []
    for i in test_.get_clusters():
        clusters.append("Кластер №" + str(i))
    return clusters
    
    