import math
import random
from sys import argv

script, p, q, N, noise, fileName = argv

p = int(p)
q = int(q)
N = int(N)
noise = float(noise)

sin = math.sin
cos = math.cos

f = open(fileName,"w+")

for i in range(N):
    t = random.random() *  2 * math.pi 

    n_noise = random.randint(0, noise * 100) / 100

    #x = math.sin(t) + 2 * math.sin(2 * t) + n_noise
    #y = math.cos(t) - 2 * math.cos(2 * t) + n_noise
    #z = - math.sin(3 * t) + n_noise

    r = cos(q * t) + 2
    
    x = r * cos(p * t) + n_noise
    y = r * sin(p * t) + n_noise
    z = - sin(q * t) + n_noise


    s = str(x) + "," + str(y) + "," + str(z) 

    if(i != N - 1):
        s += "\n"
            
    f.write(s)