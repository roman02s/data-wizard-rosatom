from functools import lru_cache

import torch
import numpy as np
from sklearn.cluster import AffinityPropagation
from transformers import AutoTokenizer, AutoModel

from .answer_clustering import AnswersClustering


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Load AutoModel from huggingface model repository
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
    tokens = sbert_tokenizer(list(texts), padding=True, truncation=True, max_length=24, return_tensors='pt').to(device)
    with torch.no_grad():
        meta = sbert_model(**tokens)
    embs = np.asarray(mean_pooling(meta, tokens['attention_mask']).to("cpu"))
    return embs

params = {
    "model" : sbert_get_embeddings,
    "clustering" : AffinityPropagation,
    "clustering_params" : {
        "random_state" : 1909,
        "affinity" : "precomputed",
    }
}

test_ = AnswersClustering(params)

@lru_cache
def get_embedding(answ):
    test_.add_answer(answ, True)
    return test_.get_answers(), test_.get_clusters()

def get_answers_clustering() -> AnswersClustering:
    return test_