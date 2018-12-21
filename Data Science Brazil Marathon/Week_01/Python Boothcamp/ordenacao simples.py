n = input()
i = input().split()
resposta = []

for _ in i:
    resposta.append(int(_))

resposta.sort()

print(' '.join(map(str, resposta)))