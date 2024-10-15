import numpy as np
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt

# Definição do dtype
dtype_student = np.dtype([
    ('work', np.int8, (5,)),          # Campo para vetor de 5 inteiros
    ('likenums', np.int8, (40,)),      # Primeiro campo para vetor de 40 inteiros
    ('dislikenums', np.int8, (40,))       # Segundo campo para vetor de 40 inteiros
])
dtype_group = np.dtype([
    ('student_number', np.int8),
    ('f_work', np.int32),
    ('f_like', np.int32),
    ('f_dislike', np.int32),
    ('calculated', np.bool)
])

students = np.zeros((40),dtype=dtype_student)
groups = np.zeros((5,40//5),dtype=dtype_group)
groups['student_number'] -= 1


def likeGraph(students):
    G = nx.DiGraph()
    # Adicionando arestas para os "likes" (gostos - verde)
    for i in range(1,40):
        for like in students[i]['likenums']:
            if like:
                G.add_edge(i, like, color='green')
    
    # Adicionando arestas para os "dislikes" (não gostos - vermelho)
    for i in range(1,40):
        for dislike in students[i]['dislikenums']:
            if dislike:
                G.add_edge(i, dislike, color='red')
    
    # Desenhando o grafo
    colors = [G[u][v]['color'] for u, v in G.edges()]
    pos = nx.circular_layout(G)  # Posição dos nós
    plt.figure(figsize=(10, 10))
    
    # Desenhando nós e arestas
    nx.draw(G, pos, node_color='lightblue', with_labels=True, edge_color=colors, node_size=700, font_size=10, arrows=True)
    
    # Mostrando o gráfico
    plt.title('Grafo de Gostos e Não Gostos entre Alunos')
    plt.show()
    
def gen_data(students,folder):
    answers = pd.read_csv("answers/ini3b.csv")
    for row in answers.iterrows():
        student = students[row[1].iloc[3]]
        for i in [0,1,2]:
            ranquing = [int(num) for sublista in [a.split(" ") for a in row[1].iloc[i+4].split(",")] for num in sublista if num.isdigit()]
            student[i][:len(ranquing)] += ranquing

def gen_random_data(students):
    works = [1,2,3,4,5]
    for i,student in enumerate(students[1:]):
        random.shuffle(works)
        student[0] += works
        aux = list(range(1,40))
        aux.pop(i)
        for j in [1,2]:
            count = 0;
            while True:
                if not len(aux) or random.random() < 0.5:
                    break
                student[j][count] = aux.pop(random.randint(0, len(aux)-1))
                count+=1;

def genpop(groups,students):
    indexes = list(range(1,40))
    for i in groups.T:
        for j in i:
            if not len(indexes):
                return
            j["student_number"] = indexes.pop(random.randint(0, len(indexes)-1))

#def fitness(groups, students):
        

gen_random_data(students)
likeGraph(students)
genpop(groups,students)

for group in groups:
    for student in group:
        others = group[group != student]["student_number"]
        likenums = students[student["student_number"]]["likenums"]
        dislikenums = students[student["student_number"]]["dislikenums"]
        student["f_like"] = np.sum(40 - np.where(np.isin(likenums, others))[0])
        student["f_dislike"] = np.sum(np.where(np.isin(dislikenums, others))[0] - 40)
        
np.sum(groups["f_like"], axis=1)