for i in range(0, 126):
    with open('numbers-lp.txt', 'a') as f:
        if i < 9:
            print('LicensePlate0000000'+ str(i+1), file=f)
        else:
           if (i >= 9) & (i < 99):
             print('LicensePlate000000'+ str(i+1), file=f)
           else:
             print('LicensePlate00000' + str(i+1), file=f)


