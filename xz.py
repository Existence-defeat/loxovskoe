#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdbool.h>
#define pi 3.14
def imp(x,y):
    return (not x) or y

def fun(x,a):
    return imp((x&73 ==0 ), imp(x&28!=0,x&a!=0))
sp = []
for a in range (1,111):
    if all(fun(x,a) for x in range (0,1300) ):
        sp.append(a)
print(min(sp))        
