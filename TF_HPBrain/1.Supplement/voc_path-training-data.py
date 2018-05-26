import os
import os.path

pathh = "/Users/chung/hpbint/gt-predicted/JPEGImages/"

for filenames in os.walk(pathh):
    filenames = list(filenames)
    filenames = filenames[2]
    for filename in filenames:
        print(filename)
        with open ("train-3c.txt",'a') as f:
            f.write(pathh+filename+'\n')
