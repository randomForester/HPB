import os
import os.path
from os import getcwd

# Training Set (Partial)

#nechk@nechk-ma-lt01:~/NECHK-Results/cars-lp-hk-all/cars-lp-hk-perfect-3$ ls -l | grep ^- | wc -l
#641

#nechk@nechk-ma-lt01:~/NECHK-Results/cars-lp-hk-all/cars-lp-hk$ ls -l | grep ^- | wc -l
#50

# 641 + 50 = 691

# Testing Set

#nechk@nechk-ma-lt01:~/NECHK-Results/cars-lp-hk-all/cars-lp-hk-perfect-1$ ls -l | grep ^- | wc -l
#190

# 692 + 190 = 882

wd = getcwd()
print(wd)

pathh = wd + "/cars-lp-hk-perfect-1/"

alist = range(692, 882)

for filenames in os.walk(pathh):
    filenames = list(filenames)
    filenames = filenames[2]
    for filename, i in zip(filenames, alist):
        print(filename)
        with open ("train+test.txt",'a') as f:
            f.write('mv ' + filename+ ' LicensePlate00000' + str(i) + '.jpg' + '\n')

