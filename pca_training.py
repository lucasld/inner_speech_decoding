import PCA.pca_Methods as pm
import PCA.pca_Models as pmod
import models.EEGNet as me
import numpy as np
import tensorflow as tf

print('--- Load data ---')

data, events = pm.load_data(subjects=range(1, 11))

PCA_TYPE = 2
PCA_COMPONENTS = 46
PCA_CONDITION = 0

TRAIN_SIZE = 0.75
TEST_SIZE = 0.25

DROPOUT = 0.4
BATCHSIZE = 15
EPOCHS = 15


model = pmod.SmallConv(10, DROPOUT)

MODELNAME = 'NPSmallConv1_i10' # 'EEGNet1_D08_Kl3'
"""
print('--- Prepare Pretraining Data ---')

preX = np.array(data['pronounced speech'])
preX = np.concatenate([preX, np.array(data['visualized condition'])], axis=0)
preY = events['pronounced speech']
preY = np.concatenate([preY, events['visualized condition']], axis=0)
preY = np.array(preY)

print('--- Pretraining ---')



#model = me.EEGNet(nb_classes=4, Chans=PCA_COMPONENTS, Samples=640, dropoutRate=0.3, kernLength=2, F1=8, D=3, F2=16,
 #                 dropoutType='Dropout')

pre_hist, pre_eval, _, path = pm.pretraining(data=(preX, preY),
                                             model=model,
                                             batchsize=BATCHSIZE, epochs=EPOCHS,
                                             train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                                             save=True, model_name=MODELNAME, filename=f'{MODELNAME}_pretrain',
                                             pca_type=PCA_TYPE, pca_components=PCA_COMPONENTS,
                                             pca_condition=PCA_CONDITION)
pm.pretrainingVisualization(pre_hist, pre_eval, epochs=EPOCHS, batchsize=BATCHSIZE, save=True,
                            name=f"{MODELNAME}_Pretraining")

print('--- Pretraining done ---') """
print('--- Prepare Training Data ---')

X = np.array(data['inner speech'])
Y = np.array(events['inner speech'])

print('--- Start k-fold Cross Validation ---')

PCA_TYPE = 2
PCA_COMPONENTS = 46
PCA_CONDITION = 0
BATCHSIZE = 10
EPOCHS = 10
FOLDS = 10
path = 'test_models/Pretraining/SmallConv1_i10_e15_bs15_vacc26_vstd0'


#data, events = pm.load_single_subj(10)
#X = np.array(data['inner speech'])
#Y = np.array(events['inner speech'])
#MODELNAME = 'SmallConv1_i10_sub'+str(10)
hist, ev, cvs, _ = pm.kFoldTraining(data=(X, Y), model=tf.keras.models.load_model(path),
                                    batchsize=BATCHSIZE, epochs=EPOCHS, folds=FOLDS,
                                    save=False, model_name='SC1_C2D+03', filename=f'{MODELNAME}_Training',
                                    pca_type=PCA_TYPE, pca_components=PCA_COMPONENTS, pca_condition=PCA_CONDITION)
pm.kFoldVisualization(hist, ev, folds=FOLDS, epochs=EPOCHS, batchsize=BATCHSIZE, save=True,
                      name=f"{MODELNAME}_Training")
print('--- k-fold Cross Validation done ---')
