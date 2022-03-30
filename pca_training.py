import PCA.pca_Methods as pm
import PCA.pca_Models as pmod
import models.EEGNet as me
import PCA.pca_utilities as pu
import numpy as np
import tensorflow as tf
from ResNet import ResNet

print('--- Load data ---')

pretraining, training = pm.load_data(subjects=range(1,2))

PCA_TYPE = 2
PCA_COMPONENTS = 46
PCA_CONDITION = 0

TRAIN_SIZE = 0.75
TEST_SIZE = 0.25

DROPOUT = 0.4
BATCHSIZE = 15
EPOCHS = 1

# Initialize the loss-function
loss = tf.keras.losses.CategoricalCrossentropy()
# Initialize the optimizer
optimizer = tf.keras.optimizers.Adam(0.001)
# Initialize Model
model = pmod.DenseConv() # ResNet(num_resBlock=[[8, 16], [16, 32]])
# Build model to output it's summary

#model = pmod.SmallConv(6, DROPOUT)

MODELNAME = 'Test' # SmallConv1_i6' # 'EEGNet1_D08_Kl3'

print('--- Pretraining ---')

#model = me.EEGNet(nb_classes=4, Chans=PCA_COMPONENTS, Samples=640, dropoutRate=0.3, kernLength=128, F1=8, D=3, F2=16,
 #                 dropoutType='Dropout')

pre_hist, pre_eval, _, path = pm.pretraining(data=pretraining,
                                             model=model, loss=loss, optimizer=optimizer,
                                             batch_size=BATCHSIZE, epochs=EPOCHS,
                                             train_size=TRAIN_SIZE, test_size=TEST_SIZE,
                                             save=True, model_name=MODELNAME, filename=f'{MODELNAME}_pretrain',
                                             pca_type=PCA_TYPE, pca_components=PCA_COMPONENTS,
                                             pca_condition=PCA_CONDITION)
pu.k_fold_visualization(pre_hist, pre_eval, epochs=EPOCHS, batch_size=BATCHSIZE, save=True,
                        name=f"{MODELNAME}", folder="figures/Pretraining")

print('--- Pretraining done ---')

print('--- Start k-fold Cross Validation ---')

PCA_TYPE = 2
PCA_COMPONENTS = 46
PCA_CONDITION = 0
BATCHSIZE = 10
EPOCHS = 1
FOLDS = 2
#path = 'test_models/Pretraining/SmallConv1_i10_e15_bs15_vacc26_vstd0'

hist, ev, cvs, _ = pm.k_fold_training(data=training,
                                      model=model, loss=loss, optimizer=optimizer,
                                      batch_size=BATCHSIZE, epochs=EPOCHS, folds=FOLDS,
                                      save=False, model_name=f'{MODELNAME}', filename=f'{MODELNAME}_Training',
                                      pca_type=PCA_TYPE, pca_components=PCA_COMPONENTS, pca_condition=PCA_CONDITION)
pu.k_fold_visualization(hist, ev, epochs=EPOCHS, batch_size=BATCHSIZE, save=True,
                        name=f"{MODELNAME}")
print('--- k-fold Cross Validation done ---')
