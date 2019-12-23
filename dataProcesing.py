import re
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
# training set 
train_labels = pd.read_csv('train_5_labels.csv', 
                            names=['ImageID','Caption'],
                            index_col=['ImageID'])

train_labels_freq = defaultdict(int)
for r in train_labels['Caption']:
    labels = r.split()
    for l in labels:
        train_labels_freq[l] += 1
train_labels_freq=sorted(train_labels_freq.items(),key=lambda item:item[1],reverse=True )

train_labels_list = [x[0] for x in train_labels_freq[:1000]]
label_2_idx = {}
idx_2_label = {}
for i,v in enumerate(train_labels_list):
    label_2_idx[v] = i
    idx_2_label[i] = v

#class_descriptions = pd.read_csv('class-descriptions.csv', index_col='label_code')
img_ids = list(train_labels.index.unique())
img_ids1, img_ids2 = train_test_split(img_ids, test_size=0.1, random_state=21)
del img_ids1
train_ids, valid_ids = train_test_split(img_ids2, test_size=0.1, random_state=21)


# test set
test_data=pd.read_csv('inclusive_images_stage_1_solutions.csv',names=['ImageID','Caption'],index_col=['ImageID'])

test_labels=test_data.values.tolist()[:1000]
test_label_list=[]
for row in test_labels:
    for label in re.split(' ',row[0]):
        if label not in test_label_list:
            test_label_list.append(label)
# Merge classes with training set             
count=len(train_labels_list)
for v in test_label_list:
    if v not in label_2_idx.keys():
        label_2_idx[v] = count
        count+=1
       
test_labels=test_data.values.tolist()[1000:2000]
test_ids = list(test_data.index.unique())[1000:2000]

print("classes:",len(label_2_idx))
print("train:",len(train_ids))