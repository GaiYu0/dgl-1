import sys
from props import penalized_logp, similarity


with open("../exp_temp/logp04_temp.txt", 'r') as file:
    for line in file:
        x,y = line.split(",")[:2]
        if y == "None": y = None
        sim = similarity(x, y)
        try:
            prop = penalized_logp(y) - penalized_logp(x)
            print (x, y, sim, prop)
        except Exception as e:
            print (x, y, sim, 0.0)

"""
for line in sys.stdin:
    x,y = line.split()[:2]
    if y == "None": y = None
    sim = similarity(x, y)
    try:
        prop = penalized_logp(y) - penalized_logp(x)
        print x, y, sim, prop
    except Exception as e:
        print x, y, sim, 0.0
"""

if __name__ == "__main__":
    print("test")