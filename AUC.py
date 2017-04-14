#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2017/4/14 9:33
# @Author  : ConanCui
# @Site    : 
# @File    : AUC.py
# @Software: PyCharm Community Edition
import numpy as np
def plotROC(predStrengths,classLabels):
    '''
    其中纵轴为真正例所占正样本的比例，即真阳率TPR = TP/(TP+FN),其中(TP+FN)=P是正样本数目,也相当与recall
    其中横轴为伪正例所占负样本的比例，即假阳率FPR = FP/(FP+TN),其中(FP+TN)=N是负样本数目
    ROC曲线含义：对不同阈值，对样本进行从新分类，分类后从新计算TPR和FPR。即ROC曲线就是真阳率和假阳率在不同分类
    阈值下的曲线图
    具体操作：首先将分类的阈值设为最低，即此时的样本全部会被预测为P，此时的TP=P,FP=N，所以TPR,FPR=1，1，然后每次抬高阈值，
    将会有一个分类强度最低的样本i，那么该样本将会被划分为负样本。如果该样本的真实标签为P，那么此时的真阳率TPR = TP'/P =(TP-1)/P
    此时的假阳率FPR = FP'/N = FP/N.
    如果该样本的真实标签为N，那么此时的真阳率TPR = TP'/P = TP/P,此时的假阳率FPR = FP'/N = (FP-1)/N

    其他评价指标：
    recall = P/(TP+FN)
    precision = TP/(TP+FP)
    accuracy = (TP+TN)/(P+N)
    F1 = 2/(1/precision + 1/recall)
    :param predStrengths:
    :param classLabels:
    :return:
    '''
    import matplotlib.pyplot as plt
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(np.array(classLabels) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist():
        if classLabels[index] ==1.0:
            delx = 0;dely =yStep
        else:
            delx = xStep;dely = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delx],[cur[1],cur[1] - dely] , c = 'b')
        cur = (cur[0]-delx , cur[1] -dely)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate');plt.ylabel('True Positive Rate')
    plt.title('Roc curve')
    ax.axis([0,1,0,1])
    plt.show()
    print ("the Area Under the Curve is: %f" % (ySum*xStep))

if __name__ == '__main__':
    predStrengths = np.random.rand(100)
    classLabels = np.random.rand(100)
    classLabels[classLabels>0.5] = 1
    classLabels[classLabels<=0.5] = 0
    plotROC(predStrengths,classLabels)
