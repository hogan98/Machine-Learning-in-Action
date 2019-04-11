
# coding: utf-8

# In[5]:


# %load My_KNN.py
'''

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
'''




from numpy import *
import operator
def createDataSet():
    groups=array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return groups, labels
createDataSet()

groups=array([[1,1.1],[1,1],[0,0],[0,0.1]])
labels=['A','A','B','B']

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistance=sqDiffMat.sum(axis=1)
    distances=sqDistance**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
classify0([0,0],groups,labels,3)


# In[6]:


def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    numberOfLines=len(arrayOLines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromline=line.split('\t')
        returnMat[index:]=listFromline[0:3]
        classLabelVector.append(int(listFromline[-1]))
        index += 1
    return returnMat,classLabelVector


# In[37]:


datingDataMat, datingLabels=file2matrix('datingTestSet2.txt')
datingDataMat, datingLabels


# In[9]:


import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax=fig.add_subplot(111)
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()


# In[16]:


def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataset=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataset=dataSet-tile(minVals,(m,1))
    normDataset=normDataset/tile(ranges,(m,1))
    return normDataset, ranges, minVals
normMat,ranges,minVals=autoNorm(datingDataMat)
normMat,ranges,minVals


# In[35]:


def datingClassTest():
    horatio=.10
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*horatio)
    errorCount=.0
    for i in range(numTestVecs):
        classfierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with:%d, the real answer is %d'%(classfierResult,datingLabels[i]))
    if (classfierResult!=datingLabels[i]):
        errorCount +=1
    print('the total error rate is: %f'%(errorCount/float(numTestVecs)))
datingClassTest() 


# In[40]:


def classifyPerson():
    resultlist=['no','small doses','large doses']
    percentTats=float(input('percentage of time spent playing games'))
    ffMiles=float(input('frequent flier miles earned per year'))
    iceCream=float(input('liters of ice cream consumed per year'))
    datingDataMat, datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    inArr=array([ffMiles,percentTats,iceCream])
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like this peroson',resultlist[classifierResult-1])
classifyPerson()   

