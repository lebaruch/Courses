n = int(input("Quantidade de questões:"))
gabarito = input("Gabarito da prova:")
respostas = input("Respostas do candidato:")

total = 0

for o, c in (zip(gabarito, respostas)):
    if o == c:
        total += 1

print("Acertos:{}".format(total))