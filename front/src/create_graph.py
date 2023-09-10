import uuid
import random

import pandas as pd

import networkx as nx

def random_color():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f'rgb({r}, {g}, {b})'
    
def create_graph(question, answers, clusters):
    # question, answer, cluster
    G = nx.DiGraph()
    quest_k = uuid.uuid4()
    G.add_node(quest_k, level=0, name_1=question, color=random_color())
    
    _cluesters = []
	# clusters
    for cluester in clusters:
        _cluesters.append(cluester)
        G.add_node(cluester, level=1, name_1=cluester, color=random_color())
        G.add_edge(quest_k, cluester)
	
 	# embeddings
    for ind, answer in enumerate(answers):
        k_answer = uuid.uuid4()
        # print(clusters, ind, clusters[ind], G.nodes[0])
        G.add_node(k_answer, level=2, name_1=answer, color=G.nodes[_cluesters[ind]]['color'])
        G.add_edge(_cluesters[ind], k_answer)
	
    return G


def add_answer(G, question, cluster, client_answer):
    print("IN ADD_ANSWER: ", question, cluster, client_answer)
    quest = None
    clust = None
    for node in G.nodes():
        if G.nodes[node]['name_1'] == question:
            quest = node
    for node in G.nodes():
        print(node, G.nodes[node])
        if G.nodes[node]['name_1'] == cluster:
            clust = node
    k_answer = uuid.uuid4()
    if clust is None:
        klust_k = uuid.uuid4()
        G.add_node(klust_k, level=1, name_1=cluster, color=random_color())
        clust = klust_k
    G.add_node(k_answer, level=2, name_1=client_answer, color=G.nodes[clust]['color'])
    G.add_edge(clust, k_answer)
    G.add_edge(quest, clust)



# def graph_to_dataframe(G):
#     # question, answer, cluster
#     df = pd.DataFrame()
#     quest = None
#     for node in G.nodes():
#         if G.nodes[node]['level'] == 0:
#             quest = G.nodes[node]['name_1']
#     for node in G.nodes():
#         if G.nodes[node]['level'] == 2:
#             df = pd.DataFrame({'question': quest, 'answer': G.nodes[node]['name_1'], 'cluster': G.successors[node]}, ignore_index=True)
#     return df


def graph_to_dataframe(G):
    # question, answer, cluster
    df = pd.DataFrame(columns=['question', 'answer', 'cluster'])
    quest = None
    for node in G.nodes():
        if G.nodes[node]['level'] == 0:
            quest = G.nodes[node]['name_1']
    print("quest", quest)
    for node in G.nodes():
        if G.nodes[node]['level'] == 2:
            print("G.nodes[node]", G.nodes[node])
            df = pd.concat([df, pd.DataFrame([{'question': quest, 'answer': G.nodes[node]['name_1'], 'cluster': node}])], ignore_index=True)
    return df


