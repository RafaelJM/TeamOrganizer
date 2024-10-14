import numpy as np
import pandas as pd

# Definição do dtype
dtype_student = np.dtype([
    ('work', np.int8, (5,)),          # Campo para vetor de 5 inteiros
    ('likenums', np.int8, (40,)),      # Primeiro campo para vetor de 40 inteiros
    ('dislikenums', np.int8, (40,))       # Segundo campo para vetor de 40 inteiros
])

students = np.zeros((40),dtype=dtype_student)
answers = pd.read_csv("answers/ini3b.csv")

for row in answers.iterrows():
    student = students[row[1].iloc[3]]
    for i in [0,1,2]:
        ranquing = [int(num) for sublista in [a.split(" ") for a in row[1].iloc[i+4].split(",")] for num in sublista if num.isdigit()]
        student[i][:len(ranquing)] += ranquing