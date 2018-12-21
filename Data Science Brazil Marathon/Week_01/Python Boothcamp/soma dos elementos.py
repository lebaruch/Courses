#Entrada de dados
n = int(input())
x = input().split()

for i in range(len(x)):
    x[i] = int(x[i])


#for loop

total = 0
for _ in x:
    total = total + _

print(total)