from keras.applications.densenet import DenseNet121
from keras.applications.densenet import preprocess_input
import cv2 as cv
from keras.models import Model, load_model
from keras.layers import *
from keras.callbacks import *
from keras.preprocessing.image import load_img, img_to_array



def caption_2_one_hot(caption, n_classes=1, lookup_dict=None):
    y = np.zeros((n_classes))
    for w in caption.split():
        if w in lookup_dict.keys():
            idx = lookup_dict[w]
            y[idx] = 1
    return y

def ImageDataGen(ids, df,lookup_dict,n_classes,
                 img_dir, input_size,
                 bs):
    x_batch = []
    y_batch = []
    for start in range(0, len(ids), bs):
        end = min(start+bs, len(ids))
        sample = ids[start:end]
        for img_id in sample:
            img = cv.imread('{}/{}.jpg'.format(img_dir, img_id))
            if img is not None:
                img = cv.resize(img, (input_size, input_size))
                img = preprocess_input(img.astype(np.float32))
                x_batch.append(img)
                caption = df.loc[img_id]['Caption']
                y = caption_2_one_hot(caption, n_classes=n_classes, lookup_dict=lookup_dict)
                y_batch.append(y)
    return np.array(x_batch, np.float32),np.array(y_batch, np.float32)

def ImageDataGen1(ids, df,lookup_dict,n_classes,input_size,
                 bs,img_dir='train_5',  returnIds=False):
    while True:
        for start in range(0, len(ids), bs):
            x_batch = []
            y_batch = []
            end = min(start+bs, len(ids))
            sample = ids[start:end]
            for img_id in sample:
                img = cv.imread('{}/{}.jpg'.format(img_dir, img_id))
                if img is not None:
                    img = cv.resize(img, (input_size, input_size))
                    img = preprocess_input(img.astype(np.float32))
                    x_batch.append(img)
                    caption = df.loc[img_id]['Caption']
                    y = caption_2_one_hot(caption, n_classes=n_classes, lookup_dict=lookup_dict)
                    y_batch.append(y)
                    
            x_batch = np.array(x_batch, np.float32)
            y_batch = np.array(y_batch, np.float32)
            
            if returnIds:
                yield x_batch, y_batch, sample
            else:
                yield x_batch, y_batch

def ClsModel(n_classes=1, input_shape=(224,224,3)):
    base_model = DenseNet121(weights=None, include_top=False, input_shape=input_shape)
    x = AveragePooling2D(pool_size=(3,3), name='avg_pool')(base_model.output)
    x = Flatten()(x)
    x = Dense(1024, activation='relu', name='dense_post_pool')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_classes, activation='softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, output=output)
    return model
