import os
import os.path
from os import getcwd

#nechk@nechk-ma-lt01:~/NECHK-Results/cars-lp-hk-all/rename-second/Data100/Data10$ ls -l | grep ^- | wc -l
#9

#mv Data10/ ..

#nechk@nechk-ma-lt01:~/NECHK-Results/cars-lp-hk-all/rename-second/Data100$ ls -l | grep ^- | wc -l
#90

#mv Data10 ../
#mv Data100 ../

#nechk@nechk-ma-lt01:~/NECHK-Results/cars-lp-hk-all/rename-second$ ls -l | grep ^- | wc -l
#780

wd = getcwd()
print(wd)

#pathh = wd + "/rename-second/Data100/Data10/"
#pathh = wd + "/rename-second/Data100/"
pathh = wd + "/rename-second/"

#alist = range(1, 10)
#alist = range(10, 100)
alist = range(100, 880)

for filenames in os.walk(pathh):
    filenames = list(filenames)
    filenames = filenames[2]
    for filename, i in zip(filenames, alist):
        print(filename)
        with open ("train+test-second.txt",'a') as f:
            f.write('mv ' + filename+ ' LicensePlate00000' + str(i) + '.jpg' + '\n')

