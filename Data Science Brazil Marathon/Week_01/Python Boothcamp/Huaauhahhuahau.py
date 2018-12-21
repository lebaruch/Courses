n = input()
vogais = ['a', 'e', 'i', 'o', 'u']

resultado = ''

for _ in n:
    if _ in vogais:
        resultado += _

if resultado == resultado[::-1]:
    print("S")
else:
    print("N")