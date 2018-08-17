import os
import os.path

pathh = "/home/cesare/Github/cars_markus-all-used/image/"

#test
filename = "example.jpeg"
filename.split(".")[-1]
filename.split(".")[0]
"filename".split(".")[-1]
#test

for filenames in os.walk(pathh):
    filenames = list(filenames)
    filenames = filenames[2]
    for filename in filenames:
        print(filename)
        with open ("train.txt",'a') as f:
            f.write(filename.split(".")[0]+'\n')
