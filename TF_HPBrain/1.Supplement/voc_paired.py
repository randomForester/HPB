'''
Annotations/xml
JPEGImages/jpg

cp Annotations/* Test/
cp JPEGImages/* Test/
'''

import os
import os.path

h = 0
a = ''
b = ''
dele = []

pathh = "/Users/chung/hpbint/gt-predicted/Test/"

#dele.remove(1)
for filenames in os.walk(pathh):
    filenames = list(filenames)
    filenames = filenames[2]
    for filename in filenames:
        
        print(filename)
        if h==0:
            a = filename
            h = 1
        elif h==1:
            #print(filename)
            b = filename
            if a[0:a.rfind('.', 1)]==b[0:b.rfind('.', 1)]:
                h = 0
            #print(filename)
            else:
                h = 1
                dele.append(a)
                a = b
    else:
        print("wa1")
print(dele)
for file in dele:
    os.remove(pathh+file)
    print("remove"+file+" is OK!")

# Repeat searching unpaired file
for filenames in os.walk(pathh):
    filenames = list(filenames)
    filenames = filenames[2]
    for filename in filenames:
        
        print(filename)
        if h==0:
            a = filename
            h = 1
        elif h==1:
            #print(filename)
            b = filename
            if a[0:a.rfind('.', 1)]==b[0:b.rfind('.', 1)]:
                h = 0
            #print(filename)
            else:
                h = 1
                dele.append(a)
                a = b
    else:
        print("wa1")
print (dele)
