import networkx as nx

import plotly.graph_objs as go
from plotly.subplots import make_subplots

def get_node_size(level):
    if level == 0:
        return 50
    elif level == 1:
        return 35
    else:
        return 15


def create_graph_viz(G: nx.Graph):
    # Получаем позиции узлов для отображения
    pos = nx.circular_layout(G)

    # Создаем список узлов и ребер для отрисовки
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',  # Обратите внимание на режим markers+text
        hoverinfo='text',
        textfont=dict(size=10, color='black')  # Настройка стиля текста
    )

    node_trace['marker']['size'] = []
    node_trace['marker']['color'] = []

    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += (x,)
        node_trace['y'] += (y,)
        # node_info = f'Узел {node}<br>Связи: {len(G.edges(node))}'
        # node_info = f'{node}'
        node_trace['text'] += (G.nodes[node]['name_1'],)
        node_trace['marker']['size'] += (get_node_size(G.nodes[node]['level']), )
        node_trace['marker']['color'] += (G.nodes[node]['color'],)

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += (x0, x1, None)
        edge_trace['y'] += (y0, y1, None)

    # Создаем подзаголовок для визуализации
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(node_trace)
    fig.add_trace(edge_trace)

    # Настраиваем внешний вид визуализации
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)'  # Устанавливаем прозрачный фон
    )

    return fig

