import numpy as np
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
import re

max_size_class = 40
groups_num = 5

# Definição do dtype
dtype_student = np.dtype([
    ('work', np.int8, (5,)),          # Campo para vetor de 5 inteiros
    ('likenums', np.int8, (max_size_class,)),      # Primeiro campo para vetor de max_size_class inteiros
    ('dislikenums', np.int8, (max_size_class,)),    # Segundo campo para vetor de max_size_class inteiros
    ('valid', np.bool)       
])

dtype_group = np.dtype([
    ('student_number', np.int8),
    ('f_work', np.int32),
    ('f_like', np.int32),
    ('f_dislike', np.int32),
    ('calculated', np.bool)
])

students = np.zeros((max_size_class),dtype=dtype_student)
groups = np.zeros((groups_num,int(np.ceil(max_size_class/groups_num))),dtype=dtype_group)
groups['student_number'] -= 1


def likeGraph(students):
    G = nx.DiGraph()
    # Adicionando arestas para os "likes" (gostos - verde)
    for i in range(1,max_size_class):
        for like in students[i]['likenums']:
            if like:
                G.add_edge(i, like, color='green')
    
    # Adicionando arestas para os "dislikes" (não gostos - vermelho)
    for i in range(1,max_size_class):
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
    
def gen_data(students,folder = "data/answers.csv"):
    answers = pd.read_csv(folder)
    for row in answers.iterrows():
        student = students[row[1].iloc[3]]
        student["valid"] = True
        for i in [0,1,2]:
            ranquing = list(map(int, re.findall(r'\d+', row[1].iloc[i+4])))
            student[i][:len(ranquing)] += ranquing

def gen_random_data(students):
    works = [1,2,3,4,5]
    for i,student in enumerate(students[1:]):
        random.shuffle(works)
        student[0] += works
        aux = list(range(1,max_size_class))
        aux.pop(i)
        for j in [1,2]:
            count = 0;
            while True:
                if not len(aux) or random.random() < 0.5:
                    break
                student[j][count] = aux.pop(random.randint(0, len(aux)-1))
                count+=1;

def genpop(groups,students):
    indexes = list(range(1,max_size_class))
    for i in groups.T:
        for j in i:
            if not len(indexes):
                return
            j["student_number"] = indexes.pop(random.randint(0, len(indexes)-1))

gen_random_data(students)
likeGraph(students)
genpop(groups,students)

#def fitness(groups, students):
for group in groups:
    for student in group:
        others = group[group != student]["student_number"]
        likenums = students[student["student_number"]]["likenums"]
        dislikenums = students[student["student_number"]]["dislikenums"]
        student["f_like"] = np.sum(max_size_class - np.where(np.isin(likenums, others))[0])
        student["f_dislike"] = np.sum(np.where(np.isin(dislikenums, others))[0] - max_size_class)
        
np.sum(groups["f_like"], axis=1)