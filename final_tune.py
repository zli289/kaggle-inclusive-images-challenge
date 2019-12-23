import numpy as np
import pandas as pd
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import os
import re
import pickle
from collections import defaultdict
import dataProcesing as data
import classifier as clf
import f2score
from keras.optimizers import Adam, SGD
from keras.callbacks import *
import tensorflow as tf
from keras.layers import *

N_CLASSES=len(data.label_2_idx)
BATCH_SIZE = 8
INPUT_SIZE = 224

#tune set
tune_data=pd.read_csv('tuning_labels.csv',names=['ImageID','Caption'],index_col=['ImageID'])
        
tune_labels=tune_data.values.tolist()[:100]
tune_ids = list(tune_data.index.unique())[:100]

x_test,y_test= clf.ImageDataGen(data.test_ids, data.test_data,data.label_2_idx,
                                N_CLASSES,'inclusive_images_stage_1_images',INPUT_SIZE, bs=BATCH_SIZE)

#model config
model = clf.ClsModel(N_CLASSES)
model.summary()
model.load_weights('densenet.20.hdf5')

model.layers.pop()
model.outputs = [model.layers[-1].output]
model.layers[-1].outbound_nodes = []
output=Dense(N_CLASSES, activation='softmax',name='predictions')
for layer in model.layers[:-2]:
    layer.trainable = False
sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

tune_gen= clf.ImageDataGen(tune_ids, tune_data,data.label_2_idx,
                                N_CLASSES,'inclusive_images_stage_1_images',INPUT_SIZE, bs=BATCH_SIZE)

model_checkpoint = ModelCheckpoint(('./densenet.{epoch:02d}.hdf5'),
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   save_weights_only=True)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=2, verbose=1)
callbacks = [model_checkpoint, reduce_learning_rate]
model.fit_generator(generator=tune_gen, 
                    epochs=25, 
                    steps_per_epoch=np.ceil(len(tune_ids)/BATCH_SIZE),
                   callbacks=callbacks)

# Score trained model.
y_pred=model.predict(x_test, batch_size=BATCH_SIZE, verbose=1)
f2=f2score.f_score(y_test, y_pred)
sess = tf.Session()
print(sess.run(f2))
sess.close()
scores = model.evaluate(x_test, y_test,batch_size=BATCH_SIZE, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])