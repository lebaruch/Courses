n = int(input())
acoes = input().split()


a = 0
b = 0

for acao in acoes:
    if(acao == '1'):
        a = 0 if a == 1 else 1

    elif acao == '2':
        a = 0 if a == 1 else 1
        b = 0 if b == 1 else 1


print(a)
print(b)