import uuid
import random

import networkx as nx

def create_graph(question, answers, clusters):
    def random_color():
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        return f'rgb({r}, {g}, {b})'
    G = nx.DiGraph()
    quest_k = uuid.uuid4()
    G.add_node(quest_k, level=0, name_1=question, color=random_color())
    
    _cluesters = []
	# clusters
    for cluester in clusters:
        t = uuid.uuid4()
        _cluesters.append(t)
        G.add_node(t, level=1, name_1=cluester, color=random_color())
        G.add_edge(quest_k, t)
	
 	# embeddings
    for ind, answer in enumerate(answers):
        k_answer = uuid.uuid4()
        # print(clusters, ind, clusters[ind], G.nodes[0])
        G.add_node(k_answer, level=2, name_1=answer, color=G.nodes[_cluesters[ind]]['color'])
        G.add_edge(_cluesters[ind], k_answer)
	
    return G
