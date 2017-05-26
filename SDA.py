# SDA数据
# 19个动作，8个对象，5个传感器
# http://archive.ics.uci.edu/ml/datasets/Daily+and+Sports+Activities#

import pandas as pd
import glob
#
# 批量读取txt并生成一个csv文件
names = ['s00','s01','s02','s03','s04','s05','s06','s07','s08',
         's10','s11','s12','s13','s14','s15','s16','s17','s18',
         's20','s21','s22','s23','s24','s25','s26','s27','s28',
         's30','s31','s32','s33','s34','s35','s36','s37','s38',
         's40','s41','s42','s43','s44','s45','s46','s47','s48']

for r in glob.glob(r'Data\SDA\a[0-9][0-9]\p[0-1]\s0[2-5].txt'):
    aind = int(r[10:12])
    # data = pd.read_table(r, sep=',', names=names)
    data = pd.read_table(r, sep=',')
    data.to_csv("SDA_part.csv", mode="a+")

def get_SDA_data():
    data = pd.read_csv('Data/SDA_part.csv')
    labels = data['activity']
    classes = list(labels.drop_duplicates())
    classNum = classes.__len__()
    features = data.drop('activity', axis=1)
    return features, labels, classes, classNum

features, activity, classes, classNum = get_SDA_data()

# 生成label
activities = ['sitting', 'standing', 'lying on back', 'lying on right', 'ascending stairs', 'descending stairs',
              'standing in an elevator still', 'moving around in an elevator', 'walking in a parking lot',
              'walking on a treadmill with a speed of 4 km/h in flat', 'walking on a treadmill with a speed of 4 km/h in 15 deg inclined positions',
              'running on a treadmill with a speed of 8 km/h', 'exercising on a stepper', 'exercising on a cross trainer',
              'cycling on an exercise bike in horizontal', 'cycling on an exercise bike in vertical', 'rowing', 'jumping',
              'playing basketball']

label = []
for i in range(19):
    ac = [activities[i] for row in range(500)]
    label.extend(ac)
label = pd.DataFrame(label, columns=['activity'])
features = features.drop('index', axis=1)
data = pd.concat([features,label], axis=1) 

data.to_csv("SDA_part.csv")

