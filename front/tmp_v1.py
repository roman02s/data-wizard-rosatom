import os
import uuid

import torch

import numpy as np
import pandas as pd


import plotly.express as px

from transformers import GPT2Tokenizer, T5ForConditionalGeneration 


import time
start = time.time()
print("SUCCESS IMPOT DETOX", time.time() - start)
start = time.time()
tokenizer_for_summary = GPT2Tokenizer.from_pretrained('ai-forever/FRED-T5-1.7B',eos_token='</s>')
print("SUCCESS IMPORT TOKENIZER SUMMARY", time.time() - start)
start = time.time()
model_for_summary = T5ForConditionalGeneration.from_pretrained('ai-forever/FRED-T5-1.7B')
print("SUCCESS IMPORT MODEL SUMMARY", time.time() - start)

# save the models to disk
# Сохранение токенизатора и модели на диск
tokenizer_for_summary.save_pretrained('tokenizer_for_summary')  # Замените 'path_to_save_tokenizer' на путь к папке, где вы хотите сохранить токенизатор
model_for_summary.save_pretrained('model_for_summary.pth')  # Замените 'path_to_save_model' на путь к папке, где вы хотите сохранить модель
