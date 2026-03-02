s = int(input())
r = int(input())
vip = int(input())

k = 0 
def f(s,r,vip,srok = 0 ):
    qw = s
    s += s*(r/100)
    s -= vip * 12

    if  s<=0:
        global k
        k = srok
        return True

    if s > qw or srok > 600:
        return False
    return f(s,r,vip,srok+12)
if   f(s,r,vip):
    print("True")
    print(k)
else:
    print("False") 
    print(0)