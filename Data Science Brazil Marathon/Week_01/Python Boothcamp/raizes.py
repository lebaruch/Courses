import math

n = input()
perguntas = input().split()

resultados = []

for _ in perguntas:
    resultados.append(float(_))


for _ in resultados:
    print('{:.4f}'.format(math.sqrt(_)))