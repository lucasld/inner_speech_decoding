import PCA.pca_Methods as pm
import PCA.pca_Models as pmod
import models.EEGNet as me
import pandas as pd
import numpy as np
import tensorflow as tf

print('--- Load data ---')
#data = pd.read_csv('dataset/preprocessed/channel_pca98_df_flat_42x640')
#label = pd.read_csv('dataset/preprocessed/label')

data, events = pm.data_prep(subjects=range(1, 3))
print('--- Prepare Pretraining Data ---')

preX = np.array(data['pronounced speech'])
preX = np.concatenate([preX, np.array(data['visualized condition'])], axis=0)
preY = events['pronounced speech']
preY = np.concatenate([preY, events['visualized condition']], axis=0)
preY = np.array(preY)

print('--- Pretraining ---')
""" Model me.EEGNet(nb_classes = 4, Chans = 42,
                   Samples =640, dropoutRate = 0.8,
                   kernLength = 2, F1 = 8, D = 3, F2 = 16,
                   dropoutType = 'Dropout')"""
print('--- Run1 ---')
pre_hist, pre_eval, _, path = pm.pretraining(data=(preX, preY),
                                               model=pmod.SmallConv(),
                                               batchsize=10, folds=10, epochs=20,
                                               save=False, model_name='sC1', filename='sC1_p1')
pm.kFoldVisualization(pre_hist, pre_eval, folds = 10, save=False, name="Pretraining_Run1_sC1")

print('--- Pretraining done ---')

print('--- Prepare Training Data ---')
X = np.array(data['inner speech'])
Y = np.array(events['inner speech'])

print('--- Start k-fold Cross Validation ---')
#path = 'test_models/Pretraining/sC1_e600_bs10_vacc32_vstd2'
hist, eval, cvs, _ = pm.kFoldTraining(data=(X, Y), model=tf.keras.models.load_model(path),
                                      batchsize=10, epochs=10, folds=10,
                                      save=False, model_name='sC1', filename='sC1_Training')
pm.kFoldVisualization(hist, eval, folds=10, epochs=10, batchsize=10, save=False, name="Training_sC1")
print('--- k-fold Cross Validation done ---')
