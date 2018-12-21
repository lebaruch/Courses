def eh_primo(x):
    if x >= 2:
        for y in range(2,x):
            if not(x % y):
                return False
    else:
        return False
    return True

x = int(input())
if eh_primo(x):
    print('S')
else:
    print('N')