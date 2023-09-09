import streamlit as st
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import networkx as nx
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# question = 324

# clusters = pd.read_csv(f"../data/{question}.clusters.csv")
# embeddings = pd.read_csv(f"../data/{question}.embeddings.csv")

# data_answers = pd.concat([clusters, embeddings], axis=1)
# print(data_answers.head())

# G = nx.DiGraph()


questions = pd.read_csv("../data/all.csv")
print(questions.head())
print(questions.columns)