#
for i in range(99):
    with open('numbers.txt', 'a') as f:
        if i < 9:
            print('00'+ str(i+1), file=f)
        else:
            print('0' + str(i+1), file=f)
#

#################
# Python basics #
#################

x = 3
print(x)

def compute_number_of_days(age):
    # this function roughly computes the number of days a person has lived
    days = age * 365
    return days

days = compute_number_of_days(22)
print(days)

def print_congratulations(age, name):
    days = compute_number_of_days(age)
    print('Hello ' + name + '! You have already lived on this planet for ' + str(days) + ' days!')

print_congratulations(41, 'Cesare')

ages_parents = [51, 52]
ages_children = [2, 4, 10]

ages_family = ages_parents + ages_children
print(ages_family)

total_age = 0
for age in ages_family:
    total_age = total_age + age
print(total_age)

ages_family = [51, 52, 2, 4, 10]
names = ['Patrick', 'Maria', 'Emma', 'Jordi', 'Vasiliy']
print(len(names))

for i in range(0, len(names)):
    current_name = names[i]
    current_age = ages_family[i]
    print_congratulations(current_age, current_name)

#9X9 Table
for i in range(2, 10):
    for j in range (1, 10):
        mul = i * j
        print('If you multiply ' + str(i) + ' by ' + str(j) + ', you get ' + str(mul))
#

#################
# Python basics #
#################

'''
#
#Combine
for i in range(99):
    with open('numbers1.txt', 'a') as f:
        if i < 9:
            print('000'+ str(i+1), file=f)
        else:
            print('00' + str(i+1), file=f)
#Combine
for i in range(99, 999):
    with open('numbers2.txt', 'a') as f:
        print('0'+ str(i+1), file=f)
#Combine
#
#cat numbers1.txt numbers2.txt > numbersALL.txt
#
'''

'''
for i in range(0, 199):
    with open('numbers.txt', 'a') as f:
        if i < 9:
            print('AA0000'+ str(i+1) + ' ' + str(-1), file=f)
        else:
           if (i >= 9) & (i < 99):
             print('AA000'+ str(i+1) + ' ' + str(-1), file=f)
           else:
             print('AA00' + str(i+1) + ' ' + str(-1), file=f)
'''

'''
for i in range(30):
    with open('numbers.txt', 'a') as f:
        print('00' + str(i+1), file=f)
'''


