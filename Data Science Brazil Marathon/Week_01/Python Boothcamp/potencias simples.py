import math

n = input().split()
lista = []

for _ in n:
    lista.append(_)

numero  = float(lista[0])
potencia = float(lista[1])

print(('{:.4f}').format(math.pow(numero, potencia)))
