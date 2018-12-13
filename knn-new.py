import csv
import numpy as np
import operator

def loadTraindata():
    ll = []
    file = open('./data/train.csv')
    lines = csv.reader(file)
    for line in lines :
        ll.append(line)
    ll.remove(ll[0])
    ll = np.array(ll)
    data = ll[:, 1:]
    label = ll[:, 0]
    return transform(toInt(data)), toInt(label)

def loadTestdata():
    ll = []
    file = open('./data/test.csv')
    lines = csv.reader(file)
    for line in lines:
        ll.append(line)
    ll.remove(ll[0])
    data = np.array(ll)
    return transform(toInt(data))

def toInt(array):
    array_data = np.mat(array)
    m, n = np.shape(array_data)
    newarray = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            newarray[i, j] = int(array_data[i, j])
    return newarray

def transform(array):
    m, n = np.shape(array)
    newarray = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            newarray[i, j] = array[i, j]/255
    return newarray

def classify(Vector_test, train_data, train_label, k):
    m, n = np.shape(train_data)
    newarray = np.tile(Vector_test, (m, 1))
    sqdistance = (newarray - train_data)**2
    sqdistance = sqdistance.sum(axis=1) #按行累加
    distance = sqdistance**0.5
    #print(distance)
    distance_index = distance.argsort() #升序排列
    #print(distance_index)
    label_list = [0,0,0,0,0,0,0,0,0,0]
    for i in range(k):
        label_list[int(train_label[0, distance_index[i]])] += 1
    #print(label_list)
    return label_list.index(max(label_list))



def saveResult(result):
    with open('result-new.csv', 'w') as myFile:
        myWriter = csv.writer(myFile)
        myWriter.writerow(["ImageId", "Label"])
        for i in result:
            tmp = []
            tmp.append(i)
            myWriter.writerow(tmp)

def recognize():
    train_data, train_label = loadTraindata()
    test_data = loadTestdata()
    m, n = np.shape(test_data)
    ll = []
    for i in range(m):
        temp = test_data[i, :]
        result = classify(temp, train_data, train_label, 5)
        ll.append(result)
        print('resut:::', result)
    saveResult(ll)





recognize()


#a = classify([1,2,3],[[1,5,1],[1,6,1],[2,1,1],[1,2,4],[2,1,1]],[0,1,1,1,4],4)
#print(a)
#a, b = loadTraindata()
#print(a.shape, b.shape)

'''
c =np.array([[1,1,1,1],[1,1,1,1]])
d = c[0,2]
print(d)
e = {'a':1,'b':4,'c':2}
d = sorted(e.items(),key = operator.itemgetter(0),reverse = True)
print(d)

#1 了解 array 和 mat 在维度上的区别
aa = np.array([1,2,3])
bb = np.mat(aa)
print(aa.shape) #(3,)
print(bb.shape) #(1,3)

#2
aaa = np.array([[1,2,3]])
bbb = np.array([1,2,3])

print(aaa.shape)
print(aaa[0][2])

print(bbb.shape)
print(bbb[0][2])
'''