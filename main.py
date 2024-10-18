import numpy as np
import pandas as pd
import random
import networkx as nx
import matplotlib.pyplot as plt
import re
import copy

max_size_class = 40
groups_num = 5

# Definição do dtype
dtype_student = np.dtype([
    ('work', np.int8, (5,)),          # Campo para vetor de 5 inteiros
    ('likenums', np.int8, (max_size_class,)),      # Primeiro campo para vetor de max_size_class inteiros
    ('dislikenums', np.int8, (max_size_class,))    # Segundo campo para vetor de max_size_class inteiros
])

dtype_group = np.dtype([
    ('student_number', np.int8),
    ('f_work', np.int32),
    ('f_like', np.int32),
    ('f_dislike', np.int32)
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
                if not len(aux) or random.random() < 0.1:
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

def calculate_fitness(groups, students):
    groups["f_work"] = 0
    groups["f_like"] = 0
    groups["f_dislike"] = 0
    for group in groups:
        group["f_work"] += 0 if len(np.unique(students[groups["student_number"]]["work"][0].T[:2].flatten())) == 5 else -1000 #nao a melhor ideia
        for student in group:
            others = group[group != student]["student_number"]
            likenums = students[student["student_number"]]["likenums"]
            dislikenums = students[student["student_number"]]["dislikenums"]
            student["f_like"] = np.sum(max_size_class - np.where(np.isin(likenums, others))[0])
            student["f_dislike"] = np.sum(np.where(np.isin(dislikenums, others))[0] - max_size_class)
        
np.sum(groups["f_like"], axis=1)

peso_like = 1
peso_work = 1
peso_displike = 10

def calcular_pesos(grupos):
    fitness = groups["f_like"]*peso_like + groups["f_work"]*peso_work + groups["f_dislike"]*peso_displike
    fitness = np.abs(fitness - np.max(fitness) - 1)
    return (fitness) / np.sum(fitness)

def selecionar_celula_via_roleta(pesos):
    # Transformar a matriz em uma lista de pesos e usar a roleta
    pesos_flat = pesos.flatten()
    idx = np.random.choice(range(len(pesos_flat)), p=pesos_flat)
    return np.unravel_index(idx, pesos.shape)

def selecionar_matriz(groups):
    pesos = calcular_pesos(groups)
    i, j = selecionar_celula_via_roleta(pesos)
    return i, j

def selecionar_segunda_celula(groups, i, j):
    # Obter os índices da linha
    idxs = np.array(range(groups.shape[0]))
    
    
    # Remover o proprio grupo
    idxs = idxs[idxs != i]

    # Escolher um índice aleatório diferente
    k = random.choice(idxs)
    l = random.choice(np.array(range(groups.shape[1])))
    return k, l

def trocar_elementos(groups, i, j, k, l):
    groups[i, j], groups[k, l] = copy.deepcopy(groups[k, l]), copy.deepcopy(groups[i, j])
 
fs = []

def simulated_annealing(best_fitness, b_grupos, temperatura_inicial=100, taxa_resfriamento=0.99, limite_minimo=1):
    temperatura = temperatura_inicial
    while temperatura > limite_minimo:
        # 1. Atualizar fitness
        calculate_fitness(groups, students)
        fitness_antigo = np.sum([groups["f_like"]*peso_like,groups["f_work"]*peso_work,groups["f_dislike"]*peso_displike])
        if fitness_antigo > best_fitness:
            b_grupos = copy.deepcopy(groups)
            best_fitness = fitness_antigo;

        # 3. Selecionar célula via roleta (integral ou noturno)
        i, j = selecionar_matriz(groups)
        
        # 4. Perturbação: escolha groups de uma segunda célula e troca
        k, l = selecionar_segunda_celula(groups, i, j)
        trocar_elementos(groups, i, j, k, l)
        
        # 5. Recalcular fitness após a troca
        calculate_fitness(groups, students)
        fitness_novo = np.sum([groups["f_like"]*peso_like,groups["f_work"]*peso_work,groups["f_dislike"]*peso_displike])
        

        # 6. Aceitação com base na temperatura
        if fitness_novo >= fitness_antigo:
            # Manter a mudança
            pass
        else:
            # Aceitar a piora com uma probabilidade que depende da temperatura
            delta_fitness = fitness_novo - fitness_antigo
            if random.random() < np.exp(delta_fitness / temperatura):
                pass  # Aceitar a mudança
            else:
                # Reverter a troca
                trocar_elementos(groups, i, j, k, l)
        
        # 7. Resfriamento
        temperatura *= taxa_resfriamento
        
        fs.append(fitness_novo)
    return [best_fitness,b_grupos]

bfs = []
best_fitness = -np.inf
b_grupos = None
best_fitness,b_grupos = simulated_annealing(best_fitness,b_grupos)
first = b_grupos
while True:
    best_fitness,b_grupos = simulated_annealing(best_fitness,b_grupos)
    print(best_fitness)
    bfs.append(best_fitness)
    if len(bfs) > 10:
        if len(set(bfs[-10:])) == 1:
            break

fs = np.array(fs)
fs_normalized = (fs - np.min(fs)) / (np.max(fs) - np.min(fs))
plt.plot(fs[fs_normalized > np.mean(fs_normalized)-0.2],linestyle='-', color='b')
plt.title("Fitness ao longo das iterações")
plt.xlabel("Iterações")
plt.ylabel("Fitness")
plt.grid(True)
plt.show()