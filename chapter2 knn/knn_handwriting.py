
# coding: utf-8

# In[21]:


# %load My_KNN
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


# In[22]:


def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect


# In[26]:


def handwritingClassTest():
    import os, sys
    hwLables=[]
    trainingFileList=os.listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLables.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s'%(fileNameStr))
    testFileList=os.listdir('testDigits')
    errorCount=.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s'%(fileNameStr))
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLables,3)
        print('the classifier came back with: %d, the real answer is: %d'%(classifierResult,classNumStr))
    if(classifierResult!=classNumStr): errorCount+=1
    print('\n the total number of errors is: %d'%errorCount)
    print('\n the tota; errpr rate is: %f'%(errorCount/mTest))
handwritingClassTest()

