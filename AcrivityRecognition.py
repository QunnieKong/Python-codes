#encoding:utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise
from sklearn import neighbors
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from munkres import Munkres
import sklearn.metrics as metrics
from sklearn import preprocessing
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import itertools

# =============读取数据==================
def get_oulu_data():
    data = pd.read_csv('Data/oulu.csv')
    # data = data.sample(n=5000, axis=0, random_state=5)
    labels = data['activity']
    classes = list(labels.drop_duplicates())
    classNum = classes.__len__()
    features = data.drop('activity', axis=1)
    return features, labels, classes, classNum


def get_unify_oulu_data():
    names = ['s0', 's1', 's2', 's3', 's4', 's5', 't0', 't1', 't2', 't3', 't4', 't5']
    test = pd.read_excel('Data_unify/oulu_test.xls', names=names)
    train = pd.read_excel('Data_unify/oulu_train.xls', names=names)
    test_labels = pd.read_excel('Data_unify/oulu_test.xls',  names=['activity'], sheetname=1)
    train_labels = pd.read_excel('Data_unify/oulu_train.xls',  names=['activity'], sheetname=1)
    test_labels = test_labels.iloc[:, 0]
    train_labels = train_labels.iloc[:, 0]
    classes = list(test_labels.drop_duplicates())
    class_num = classes.__len__()
    return test, test_labels, train, train_labels, classes, class_num


def get_SDA_data():
    data = pd.read_csv('Data/SDA_part.csv')
    # data = data.sample(n=500, axis=0, random_state=5)
    data = data.drop('index', axis=1)
    # 去掉全0数据
    data = data.drop_duplicates()
    labels = data['activity']
    classes = list(labels.drop_duplicates())
    classNum = classes.__len__()
    features = data.drop('activity', axis=1)
    return features, labels, classes, classNum


def get_unify_SDA_data():
    names = ['s00', 's01', 's02', 's03', 's04', 's05', 's06', 's07', 's08',
             's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18',
             's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28',
             's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38',
             's40', 's41', 's42', 's43', 's44', 's45', 's46', 's47', 's48']
    test = pd.read_excel('Data_unify/sda_test.xls', names=names)
    train = pd.read_excel('Data_unify/sda_train.xls', names=names)
    test_labels = pd.read_excel('Data_unify/sda_test.xls', names=['activity'], sheetname=1)
    train_labels = pd.read_excel('Data_unify/sda_train.xls', names=['activity'], sheetname=1)
    test_labels = test_labels.iloc[:, 0]
    train_labels = train_labels.iloc[:, 0]
    classes = list(test_labels.drop_duplicates())
    class_num = classes.__len__()
    return test, test_labels, train, train_labels, classes, class_num


def get_OPP_ADL_data():
    data = pd.read_csv('Data/OPP_S1_ADL1.csv')
    # data = data.sample(n=2000, axis=0, random_state=5)
    features = data.drop('activity', axis=1)
    labels = data['activity']
    classes = list(labels.drop_duplicates())
    classNum = classes.__len__()
    return features, labels, classes, classNum


def get_OPP_Drill_data():
    data = pd.read_csv('Data/OPP_S1_Drill.csv')
    data = data.sample(n=5000, axis=0, random_state=1)
    features = data.drop('activity', axis=1)
    labels = data['activity']
    classes = list(labels.drop_duplicates())
    classNum = classes.__len__()
    return features, labels, classes, classNum


# =============处理数据==================
# 划分测试集和训练集
# 将数据随机分成三部分,分别用来：
# 1. training the static classifier
# 2. transfer learning phase to train the dynamic sensor
# 3. for measuring accuracy of framework
def data_devide(features, labels, rate1, rate2, source_ind, target_ind, source_dim, target_dim):
    s1 = source_ind * source_dim
    s2 = (source_ind + 1) * source_dim
    t1 = target_ind * target_dim
    t2 = (target_ind + 1) * target_dim
    data1, data_test, label1, label_test = train_test_split(features, labels, test_size=rate1, random_state=4)
    data2, data3, label2, label3 = train_test_split(data_test, label_test, test_size=rate2, random_state=4)
    source_train = data1.iloc[:, s1:s2]
    source_test = data2.iloc[:, s1:s2]
    target_train = data2.iloc[:, t1:t2]
    target_test = data3.iloc[:, t1:t2]
    source_test2 = data3.iloc[:, s1:s2]
    return source_train, source_test, source_test2, target_train, target_test, label1, label2, label3


# 标准化
def data_unify(data):
    scaler = preprocessing.StandardScaler().fit(data)
    data = scaler.transform(data)
    data = pd.DataFrame(data)
    return data


# 计算类别中心点
# 计算方法：密度阈值去噪，求取密度大于阈值dt的全部数据点均值作为类别中心点
# 【针对各类别距离不同】阈值自适应方法：排序覆盖前60%数据点
def get_cluster_center(data, labels, classes, class_num):
    codata = pd.concat([data, labels], axis=1)
    dim = data.shape[1]
    centers = np.zeros([class_num, dim])
    ite = 0

    # # Kmeans
    # clf = KMeans(n_clusters=class_num).fit(data)
    # centers = clf.cluster_centers_

    for i in classes:
        center = np.zeros([1, dim])
        subset = codata[codata['activity'] == i]
        subset.fillna('0')
        subset = subset.drop('activity', axis=1).as_matrix()
        subset[np.isfinite(subset) == True] = 0
        subset= Imputer().fit_transform(subset)
        inum = subset.shape[0]

        # 类别均值
        for ins in range(inum):
            center += subset[ins,:]
        center = [cc / inum for cc in center]
        center = np.array(center)
        centers[ite] = center

        # 类别距离中心
        # distance_matrix = pairwise.pairwise_distances(subset)
        # 用均值作为阈值
        # dt = distance_matrix.mean()
        # total = 0
        # for j in range(inum):
        #     for k in range(j, inum):
        #         if distance_matrix[j, k] >= dt:
        #             total += 1
        #             center += subset[j, :]
        # # center = [cc / total for cc in center]
        # centers[ite] = center[0]
        ite = ite + 1
    return centers


# 计算类别中心点距离编码
def get_dis_vec(cluster_centers, data, n):
    dis = []
    dis_vec = []
    for i in cluster_centers:
        dist = np.sqrt(np.sum(np.square(i - data)))
        dis.append(dist)
    sortind = np.argsort(dis)
    for i in range(n):
        dis_vec.append(sortind[i])
    for i in range(n):
        dis_vec.append(dis[sortind[i]])
    return dis_vec


# 获取有距离向量编码的targetData
def get_new_target_data(cluster_centers, source, target, n):
    ntd = []
    sdata = source.as_matrix()
    tdata = target.as_matrix()
    for i in range(0, tdata.__len__()):
        ori = list(tdata[i])
        dis_vec = get_dis_vec(cluster_centers, sdata[i], n)
        ori.extend(dis_vec)
        ntd.append(ori)
    ntd = pd.DataFrame(ntd)
    return ntd


# 将预测结果与值对应
def get_predict(inds, classes, len):
    pre = []
    for i in range(0,len):
        pre.append(classes[inds[i]])
    pd.DataFrame(pre, columns=['activity'])
    return pre


# 生成semi-label
def get_semi_label(cm, clnum):
    semi_label = []
    for i in range(0,clnum):
        label = []
        for j in range(0,clnum):
            label.append(0)
            if cm[i][j] != 0:
                label[j] += cm[i][j]
        semi_label.append(label)
    return semi_label


# 创建WLG
def get_wlg(class_num, l, classes, semi_label, source_predict, target_predict):
    wlg = [[1 for col in range(class_num)] for row in range(class_num)]
    for i in range(0, l):
        ci = classes.index(target_predict[i])
        slt = semi_label[classes.index(source_predict[i])]
        for j in range(0, class_num):
            if slt[j] != 0:
                wlg[ci][j] -= 1
    return wlg


# Confusion Matrix画图
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Naive算法
def naive(source_train, target_test, test_labels, train_labels):
    # 训练Source KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(source_train, train_labels)
    # 获取直接分类结果
    target_predict = clf.predict(target_test)
    # 评价参数
    accuracy = metrics.accuracy_score(test_labels, target_predict)
    recall = metrics.recall_score(test_labels, target_predict, average='weighted')
    f1 = metrics.f1_score(test_labels, target_predict, average='weighted')
    precision = metrics.precision_score(test_labels, target_predict, average='weighted')
    print("Naive:", accuracy, recall, f1, precision)
    # return accuracy, recall, f1, precision


# System Supervised算法
def system_supervised(source_train, source_test, target_train, target_test, label1, label2, label3):
    # 训练Source KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(source_train, label1)
    source_predict = clf.predict(source_test)

    # 训练Target KNN分类器
    tclf = neighbors.KNeighborsClassifier()
    tclf.fit(target_train, source_predict)
    # Recognition
    target_predict = tclf.predict(target_test)

    # Labelling 评价参数
    lpre = tclf.predict(target_train)
    accuracy = metrics.accuracy_score(label2, lpre)
    recall = metrics.recall_score(label2, lpre, average='macro')
    f1 = metrics.f1_score(label2, lpre, average='weighted')
    precision = metrics.precision_score(label2, lpre, average='weighted')
    print("System Supervised Labelling:", accuracy, recall, f1, precision)

    # Recognition 评价参数
    accuracy = metrics.accuracy_score(label3, target_predict)
    recall = metrics.recall_score(label3, target_predict, average='macro')
    f1 = metrics.f1_score(label3, target_predict, average='weighted')
    precision = metrics.precision_score(label3, target_predict, average='weighted')
    print("System Supervised Recognition:", accuracy, recall, f1, precision)
    # return accuracy, recall, f1, precision


# 中心点距离改进的System Supervised算法
def dis_system_supervised(source_train, source_test, source_test2, target_train, target_test, label1, label2, label3, class_num, classes, n):
    # 训练Source KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(source_train, label1)
    # 获取Source_predict结果
    source_predict = clf.predict(source_test)
    # 计算Source数据距离矩阵
    cluster_centers = get_cluster_center(source_train, label1, classes, class_num)
    new_target_train = get_new_target_data(cluster_centers, source_test, target_train, n)
    # 训练Target KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(new_target_train, source_predict)
    new_target_test = get_new_target_data(cluster_centers, source_test2, target_test, n)

    # Labelling 评价参数
    accuracy = metrics.accuracy_score(label2, source_predict)
    recall = metrics.recall_score(label2, source_predict, average='macro')
    f1 = metrics.f1_score(label2, source_predict, average='weighted')
    precision = metrics.precision_score(label2, source_predict, average='weighted')
    print("Dis-System Supervised Labelling:", accuracy, recall, f1, precision)

    # Recognition评价参数
    target_predict = clf.predict(new_target_test)
    accuracy = metrics.accuracy_score(label3, target_predict)
    recall = metrics.recall_score(label3, target_predict, average='macro')
    f1 = metrics.f1_score(label3, target_predict, average='weighted')
    precision = metrics.precision_score(label3, target_predict, average='weighted')
    print("Dis-Sys Recognition:", accuracy, recall, f1, precision)
    # return accuracy, recall, f1, precision


# PNL算法
def pnl(source_train, source_test, target_train, target_test, label1, label2, label3, class_num, classes):
    # 训练Source KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(source_train, label1)
    source_predict = clf.predict(source_train)
    # 获得混淆矩阵
    cm = confusion_matrix(label1, source_predict, classes)
    source_predict = clf.predict(source_test)
    # 获得Semi-label
    semi_label = get_semi_label(cm, class_num)
    # Target数据KMeans聚类
    len2 = target_train.shape[0]
    target_cluster = KMeans(n_clusters=class_num).fit(target_train)
    target_predict = target_cluster.labels_
    target_predict = get_predict(target_predict, classes, len2)
    # 获取WLG
    wlg = get_wlg(class_num, len2, classes, semi_label, source_predict, target_predict)
    # 匈牙利算法
    munk = Munkres()
    indexes = munk.compute(wlg)

    # Labelling
    final_predict = target_cluster.labels_
    result = []
    for i in range(0, len2):
        cInd = final_predict[i]
        result.append(classes[indexes[cInd][1]])
    # 评价参数
    accuracy = metrics.accuracy_score(label2, result)
    recall = metrics.recall_score(label2, result, average='macro')
    f1 = metrics.f1_score(label2, result, average='weighted')
    precision = metrics.precision_score(label2, result, average='weighted')
    print("PNL Labelling:", accuracy, recall, f1, precision)

    # Recognition
    len3 = target_test.shape[0]
    final_predict = target_cluster.predict(target_test)
    result = []
    for i in range(0, len3):
        cInd = final_predict[i]
        result.append(classes[indexes[cInd][1]])
    # 评价参数
    accuracy = metrics.accuracy_score(label3, result)
    recall = metrics.recall_score(label3, result, average='macro')
    f1 = metrics.f1_score(label3, result, average='weighted')
    precision = metrics.precision_score(label3, result, average='weighted')
    print("PNL-Recognition:", accuracy, recall, f1, precision)
    # return accuracy, recall, f1, precision


# 中心点距离改进的pnl算法
def dis_pnl(source_train, source_test, target_train, target_test, label1, label2, label3, class_num, classes, n):
    # 训练Source KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(source_train, label1)
    source_predict = clf.predict(source_train)
    # 获得混淆矩阵
    cm = confusion_matrix(label1, source_predict, classes)
    source_predict = clf.predict(source_test)
    # 获得Semi-label
    semi_label = get_semi_label(cm, class_num)
    # 计算Source数据距离矩阵
    cluster_centers = get_cluster_center(source_train, label1, classes, class_num)
    target_train = get_new_target_data(cluster_centers, source_test, target_train, n)
    # Target数据KMeans聚类
    len2 = target_train.shape[0]
    target_cluster = KMeans(n_clusters=class_num).fit(target_train)
    target_predict = target_cluster.labels_
    target_predict = get_predict(target_predict, classes, len2)
    # 获取WLG
    wlg = get_wlg(class_num, len2, classes, semi_label, source_predict, target_predict)
    # 匈牙利算法
    munk = Munkres()
    indexes = munk.compute(wlg)

    # Labelling 评价参数
    target_predict = target_cluster.labels_
    labelling_result = []
    for i in range(0, len2):
        cInd = target_predict[i]
        labelling_result.append(classes[indexes[cInd][1]])
    accuracy = metrics.accuracy_score(label2, labelling_result)
    recall = metrics.recall_score(label2, labelling_result, average='macro')
    f1 = metrics.f1_score(label2, labelling_result, average='weighted')
    precision = metrics.precision_score(label2, labelling_result, average='weighted')
    print("Dis-PNL Labelling:", accuracy, recall, f1, precision)

    # 识别结果
    target_test = get_new_target_data(cluster_centers, source_test2, target_test, n)
    final_predict = target_cluster.predict(target_test)
    len3 = target_test.shape[0]
    recognition_result = []
    for i in range(0, len3):
        cInd = final_predict[i]
        recognition_result.append(classes[indexes[cInd][1]])
    target_predict = recognition_result
    # 评价参数
    accuracy = metrics.accuracy_score(label3, recognition_result)
    recall = metrics.recall_score(label3, recognition_result, average='macro')
    f1 = metrics.f1_score(label3, recognition_result, average='weighted')
    precision = metrics.precision_score(label3, recognition_result, average='weighted')
    print("Dis-PNL Recognition:", accuracy, recall, f1, precision)
    # return accuracy, recall, f1, precision


# SDVL算法
def SDVL(source_train, target_train, target_test, label1, label2, label3, class_num, classes, alph, sig):
    # 训练Source KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(source_train, label1)
    source_predict = clf.predict(source_train)
    # 获得混淆矩阵
    cm = confusion_matrix(label1, source_predict, classes)
    # 获得Semi-label
    semi_label = get_semi_label(cm, class_num)
    # 计算target KNN Graph
    len2 = target_train.shape[0]
    knngraph = neighbors.kneighbors_graph(target_train, 5, p=2, mode='distance')
    # kdata = knngraph.data
    # kindices = knngraph.indices
    # kindptr = knngraph.indptr
    # 获取权值矩阵C
    # eigh = np.zeros([len3, len3])
    # for i in range(len3):
    #     bas = 0
    #     knum = kindptr[i+1]-kindptr[i]
    #     for j in range(knum):
    #         ind = kindptr[i]+j
    #         dis = kdata[ind]
    #         eigh[i, kindices[ind]] = np.exp(-1 * dis / (2 * sig * sig))
    #         bas += eigh[i, kindices[ind]]
    #     for j in range(knum):
    #         ind = kindptr[i] + j
    #         eigh[i, kindices[ind]] = eigh[i, kindices[ind]] / bas

    # 获取权值矩阵
    source_predict = clf.predict(source_test)
    eigh = np.zeros([len2, len2])
    for i in range(len2):
        bas = 0
        for j in range(len2):
            dis = knngraph[i, j]
            if dis != 0:
                eigh[i, j] = np.exp(-1 * dis / (2 * sig * sig))
                bas += eigh[i, j]
        for j in range(len2):
            if eigh[i, j] != 0:
                eigh[i, j] = eigh[i, j] / bas
    # 迭代更新
    # 原始semi-label状态矩阵
    for ite in range(2):
        ori_semi = np.zeros([len2, class_num])
        for i in range(len2):
            # 旧semi-label
            pre = source_predict[i]
            ind = classes.index(pre)
            semi = semi_label[ind]
            ori_semi[i] = semi
            ori = np.array(semi)
            fn = np.zeros([1, class_num])
            # 新semi-label
            for j in range(len2):
                if eigh[i, j] != 0:
                    pre = source_predict[j]
                    ind = classes.index(pre)
                    semij = semi_label[ind]
                    semij = np.array(semij)
                    fn = fn + alph * semij * eigh[i, j]
            ori = (1-alph) * ori + fn
            sortind = np.argsort(ori)
            maxind = sortind[0][class_num-1]
            new_pre = classes[maxind]
            source_predict[i] = new_pre
    # 评价参数
    # a = range(classes.__len__())
    # cm = confusion_matrix(source_predict, label3, classes)
    # plot_confusion_matrix(cm, a)

    # Labelling
    accuracy = metrics.accuracy_score(label2, source_predict)
    recall = metrics.recall_score(label2, source_predict, average='macro')
    f1 = metrics.f1_score(label2, source_predict, average='weighted')
    precision = metrics.precision_score(label2, source_predict, average='weighted')
    print("SDVL Labelling:", accuracy, recall, f1, precision)

    # Recognition
    clf = neighbors.KNeighborsClassifier()
    clf.fit(target_train, source_predict)
    target_predict = clf.predict(target_test)
    accuracy = metrics.accuracy_score(label3, target_predict)
    recall = metrics.recall_score(label3, target_predict, average='macro')
    f1 = metrics.f1_score(label3, target_predict, average='weighted')
    precision = metrics.precision_score(label3, target_predict, average='weighted')
    print("SDVL Recognition:", accuracy, recall, f1, precision)

    # return accuracy, recall, f1, precision


# Dis-SDVL算法
def Dis_SDVL(source_train, source_test, source_test2, target_train, target_test, label1, label2, label3, class_num, classes, alph, sig, n):
    # 训练Source KNN分类器
    clf = neighbors.KNeighborsClassifier()
    clf.fit(source_train, label1)
    source_predict = clf.predict(source_train)
    # 获得混淆矩阵
    cm = confusion_matrix(label1, source_predict, classes)
    # 获得Semi-label
    semi_label = get_semi_label(cm, class_num)
    # 计算Source数据距离矩阵
    cluster_centers = get_cluster_center(source_train, label1, classes, class_num)
    target_train = get_new_target_data(cluster_centers, source_test, target_train, n)
    # 计算target KNN Graph
    len2 = target_train.shape[0]
    knngraph = neighbors.kneighbors_graph(target_train, 5, p=2, mode='distance')
    # 获取权值矩阵
    source_predict = clf.predict(source_test)
    eigh = np.zeros([len2, len2])
    for i in range(len2):
        bas = 0
        for j in range(len2):
            dis = knngraph[i, j]
            if dis != 0:
                eigh[i, j] = np.exp(-1 * dis / (2 * sig * sig))
                bas += eigh[i, j]
        for j in range(len2):
            if eigh[i, j] != 0:
                eigh[i, j] = eigh[i, j] / bas
    # 迭代更新
    # 原始semi-label状态矩阵
    for ite in range(2):
        ori_semi = np.zeros([len2, class_num])
        for i in range(len2):
            # 旧semi-label
            pre = source_predict[i]
            ind = classes.index(pre)
            semi = semi_label[ind]
            ori_semi[i] = semi
            ori = np.array(semi)
            fn = np.zeros([1, class_num])
            # 新semi-label
            for j in range(len2):
                if eigh[i, j] != 0:
                    pre = source_predict[j]
                    ind = classes.index(pre)
                    semij = semi_label[ind]
                    semij = np.array(semij)
                    fn = fn + alph * semij * eigh[i, j]
            ori = (1-alph) * ori + fn
            sortind = np.argsort(ori)
            maxind = sortind[0][class_num-1]
            new_pre = classes[maxind]
            source_predict[i] = new_pre
    # 评价参数
    # a = range(classes.__len__())
    # cm = confusion_matrix(source_predict, label3, classes)
    # plot_confusion_matrix(cm, a)

    # Labelling
    accuracy = metrics.accuracy_score(label2, source_predict)
    recall = metrics.recall_score(label2, source_predict, average='macro')
    f1 = metrics.f1_score(label2, source_predict, average='weighted')
    precision = metrics.precision_score(label2, source_predict, average='weighted')
    print("Dis-SDVL Labelling:", accuracy, recall, f1, precision)

    # Recognition
    target_test = get_new_target_data(cluster_centers, source_test2, target_test, n)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(target_train, source_predict)
    source_predict = clf.predict(target_test)

    accuracy = metrics.accuracy_score(label3, source_predict)
    recall = metrics.recall_score(label3, source_predict, average='macro')
    f1 = metrics.f1_score(label3, source_predict, average='weighted')
    precision = metrics.precision_score(label3, source_predict, average='weighted')
    print("Dis-SDVL Recognition:", accuracy, recall, f1, precision)
    # return accuracy, recall, f1, precision

# LP
def LP(source_train, target_test, label1, label3):
    label_prop_model = LabelPropagation()
    label_prop_model.fit(source_train, label1)
    source_predict = label_prop_model.predict(target_test)
    # 评价参数
    accuracy = metrics.accuracy_score(label3, source_predict)
    recall = metrics.recall_score(label3, source_predict, average='weighted')
    f1 = metrics.f1_score(label3, source_predict, average='weighted')
    precision = metrics.precision_score(label3, source_predict, average='weighted')
    print("LP:", accuracy, recall, f1, precision)
    return accuracy, recall, f1, precision

# Recognition Upper Bound
def UB(source_test, source_test2, target_train, label2, label3):
    train = pd.concat([source_test,target_train], axis=1)
    clf = neighbors.KNeighborsClassifier()
    clf.fit(train, label2)
    test = pd.concat([source_test2,target_test], axis=1)
    predict = clf.predict(test)

    accuracy = metrics.accuracy_score(label3, predict)
    recall = metrics.recall_score(label3, predict, average='macro')
    f1 = metrics.f1_score(label3, predict, average='weighted')
    precision = metrics.precision_score(label3, predict, average='weighted')
    print("Upper Bound:", accuracy, recall, f1, precision)
    # return accuracy, recall, f1, precision
# 读取数据
# features, labels, classes, classNum = get_oulu_data()
# features, labels, classes, classNum = get_SDA_data()
# features, labels, classes, classNum = get_OPP_ADL_data()
features, labels, classes, classNum = get_OPP_Drill_data()
print(features.shape)

# # 归一化
# features = data_unify(features)
#
# # 划分训练集和测试集
source_train, source_test, source_test2, target_train, target_test, label1, label2, label3 = \
    data_devide(features, labels, 0.6, 0.3, 1, 0, 13, 13)

# 算法结果
# Naive
# naive(source_train, target_test, label3, label1)
# PNL
pnl(source_train, source_test, target_train, target_test, label1, label2, label3, classNum, classes)
# Dis-PNL
dis_pnl(source_train, source_test, target_train, target_test, label1, label2, label3, classNum, classes, 2)
# Syster Supervised
system_supervised(source_train, source_test, target_train, target_test, label1, label2, label3)
# Dis-Syster Supervised
dis_system_supervised(source_train, source_test, source_test2, target_train, target_test, label1, label2, label3, classNum, classes, 2)
# SDVL
SDVL(source_train, target_train, target_test, label1, label2, label3, classNum, classes, 0.75, 0.01)
# Dis-SDVL
Dis_SDVL(source_train, source_test, source_test2, target_train, target_test, label1, label2, label3, classNum, classes, 0.75, 0.01, 2)
# LP
# a,r,f,p =LP(source_train, target_test, label1, label3)
# Upper Bound
UB(source_test, source_test2, target_train, label2, label3)
