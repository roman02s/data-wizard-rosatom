import numpy as np
import pandas as pd

import uuid
import random

import streamlit as st
import networkx as nx


from src.create_graph import create_graph
from src.visualizazation import create_graph_viz

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
    print(create_graph(question, answers, clusters))
    
