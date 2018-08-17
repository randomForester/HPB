import os
import os.path

pathh = "/home/cesare/Github/cars_markus-all-used/image/"

for filenames in os.walk(pathh):
    filenames = list(filenames)
    filenames = filenames[2]
    for filename in filenames:
        print(filename)
        with open ("train-lp.txt",'a') as f:
            f.write(pathh+filename+'\n')
