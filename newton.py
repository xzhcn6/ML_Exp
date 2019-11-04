
def Newton_2(c):
    t = 1.0
    while abs(t*t - c) > 1e-6:
        t = (t+c/t)/2
    return t

# f(t) = t^3 -2
def Newton_3(c, t):
    while abs(t*t*t - c) > 1e-6:
        t = (2*t)/3+c/(3*t*t)
    return t

print(Newton_3(2,1))