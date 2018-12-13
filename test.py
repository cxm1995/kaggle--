import csv
import numpy as np
def loaddata():
    ll = []
    file = open('result-new.csv')
    lines = csv.reader(file)
    for line in lines :
        print(line)
        ll.append(line[0])
    print(len(ll))
    #data = np.array(ll)
    return ll

def writeresult(result):
    with open('result-new-1.csv', 'w') as myFile:
        myWriter=csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        j = 1
        for i in result:
            tmp=[]
            tmp.append(j)
            tmp.append(i)
            myWriter.writerow(tmp)
            j += 1

result = loaddata()
writeresult(result)