a, b, c = input().split()

a = int(a)
b = int(b)
c = int(c)

if a != b and a != c:
    print("A")
elif b != a and b != c:
    print("B")
elif c!= a and c != b:
    print("C")
else:
    print("*")