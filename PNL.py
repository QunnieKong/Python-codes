# python实现的plug-n-learn代码
# 包括plug-n-learn方法，System supervised方法和Naive方法
# 参考文献：
# [1]Rokni S A, Ghasemzadeh H. Plug-n-learn: automatic learning of computational algorithms 
# in human-centered internet-of-things applications[C]// Design Automation Conference. ACM, 2016:139.

import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from munkres import Munkres
from sklearn import preprocessing


# 读取数据
def get_data():
    data = pd.read_csv('data.csv')
    labels = data['activity']
    classes = list(labels.drop_duplicates())
    classNum = classes.__len__()
    features = data.drop('activity', axis=1)
    return features, labels, classes, classNum


# 划分测试集和训练集
def data_devide(features, labels, test_rate, source_num, target_num):
    s1 = source_num * 6
    s2 = (source_num + 1) * 6
    t1 = target_num * 6
    t2 = (target_num + 1) * 6
    data_train, data_test, label_train, label_test = train_test_split(features, labels, test_size=test_rate, random_state=50)
    source_train = data_train.iloc[:, s1:s2]
    source_test = data_test.iloc[:, s1:s2]
    target_train = data_train.iloc[:, t1:t2]
    target_test = data_test.iloc[:, t1:t2]
    test_len = label_test.__len__()
    train_len = label_train.__len__()
    return source_train, source_test, target_test, target_train, label_train, label_test, test_len, train_len


# 计算正确率
def get_accuracy(data_true, data_predict, l):
    hitnum = 0
    j=-1
    for i in data_true.index:
        j = j+1
        if data_true[i] == data_predict[j]:
            hitnum = hitnum + 1
    return hitnum/l


# 生成semi-label
def get_semi_label(cm, clnum):
    semi_label = []
    for i in range(0,clnum):
        label = []
        for j in range(0,clnum):
            label.append(0)
            if cm[i][j] >= 100:
                label[j] = 1
        semi_label.append(label)
    return semi_label


# 将预测结果与值对应
def get_predict(inds, classes, len):
    pre = []
    for i in range(0,len):
        pre.append(classes[inds[i]])
    pd.DataFrame(pre, columns=['activity'])
    return pre


# 创建WLG
def get_wlg(class_num, l, classes, semi_label, source_predict, target_predict):
    wlg = [[0 for col in range(class_num)] for row in range(class_num)]
    for i in range(0, l):
        ci = classes.index(target_predict[i])
        slt = semi_label[classes.index(source_predict[i])]
        for j in range(0, class_num):
            if slt[j] != 0:
                wlg[ci][j] -= 1
    return wlg


def naive():
    # 获取数据
    features, labels, classes, classNum = get_data()

    # 划分训练集和测试集
    # source选取right knee, target选取necklace
    source_train, source_test, target_test, target_train, label_train, label_test, test_len, train_len = data_devide(
        features, labels, 0.5, 1, 0)

    # 训练Source KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(source_train, label_train)


    # 获取直接分类结果
    target_predict = clf.predict(target_test)

    final_accuracy = get_accuracy(label_test, target_predict, test_len)
    print("Naive Target labelling正确率：", final_accuracy)
    tcm = confusion_matrix(label_test, target_predict, classes)
    # print(tcm)


def system_supervised():
    # 获取数据
    features, labels, classes, classNum = get_data()

    # 划分训练集和测试集
    # source选取right knee, target选取necklace
    source_train, source_test, target_test, target_train, label_train, label_test, test_len, train_len = data_devide(
        features, labels, 0.5, 1, 0)

    # 训练Source KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(source_train, label_train)

    # 获取source单独训练正确率
    source_predict = clf.predict(source_test)
    source_cluster_accuracy = get_accuracy(label_test, source_predict, test_len)
    # print("source KNN预测正确率", source_cluster_accuracy)

    # 获得混淆矩阵
    # cm = confusion_matrix(label_test, source_predict, classes)
    # print("Source KNN混淆矩阵：\n", cm)

    # 训练Target KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(target_test, source_predict)
    target_predict = clf.predict(target_test)

    final_accuracy = get_accuracy(label_test, target_predict, test_len)
    print("Systerm Supervised Target labelling正确率：", final_accuracy)
    tcm = confusion_matrix(label_test, target_predict, classes)
    # print(tcm)


def pnl():
    # 获取数据
    features, labels, classes, class_num = get_data()

    # 划分训练集和测试集
    # source选取right knee, target选取necklace
    source_train, source_test, target_test, target_train, label_train, label_test, test_len, train_len = data_devide(
        features, labels, 0.5, 1, 0)

    # 训练Source KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(source_train, label_train)

    # 获取source单独训练正确率
    source_predict = clf.predict(source_test)
    source_cluster_accuracy = get_accuracy(label_test, source_predict, test_len)
    print("Source KNN预测正确率", source_cluster_accuracy)

    # 获得混淆矩阵
    cm = confusion_matrix(label_test, source_predict, classes)
    # print("Source KNN混淆矩阵：\n", cm)

    # 获得Semi-label
    semi_label = get_semi_label(cm, class_num)

    # Target数据KMeans聚类
    target_cluster = KMeans(n_clusters=17).fit(target_test)
    target_predict = target_cluster.labels_
    target_predict = get_predict(target_predict, classes, test_len)

    # 获取WLG
    wlg = get_wlg(class_num, test_len, classes, semi_label, source_predict, target_predict)

    # 匈牙利算法
    munk = Munkres()
    indexes = munk.compute(wlg)
    result = []
    for i in range(0, test_len):
        ci = target_predict[i]
        cInd = classes.index(ci)
        result.append(classes[indexes[cInd][1]])

    final_accuracy = get_accuracy(label_test, result, test_len)
    print("PNL Target labelling正确率：", final_accuracy)
    tcm = confusion_matrix(label_test, target_predict, classes)
    return wlg, tcm


pnl()
system_supervised()
naive()
