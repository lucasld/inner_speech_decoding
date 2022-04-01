from models.eegnet import EEGNet
import pca.pca_methods as pm
import models.pca_models as pmod
import pca.pca_utilities as pu
import tensorflow as tf

# Paramters for Pretraining and Testing
PCA_TYPE = 2  # 1
PCA_COMPONENTS = 46  # 1319
PCA_CONDITION = 0

# Parameters for Pretraining
TRAIN_SIZE = 0.75
BATCH_SIZE = 15
EPOCHS = 50
PRETRAINING = True

# Parameters for Testing
BATCH_SIZE_TRAIN = 10
EPOCHS_TRAIN = 30
FOLDS = 10
TRAINING = True
PRETRAINED = True
INDIV_SUBJECTS = True
path = "pca/pretraining_models/SimpleConv_e50_bs15_vacc24"

# Model Parameters
DROPOUT = 0.3  # Conv # 0.4 # EEg
MODEL_NAME = 'SimpleConv_Random'  # SmallConv1_i6' # 'EEGNet1_D08_Kl3'
LOSS = tf.keras.losses.CategoricalCrossentropy()
OPTIMIZER = tf.keras.optimizers.Adam(0.002)

# Initialize Model

simple = pmod.SimpleConv([64,16], DROPOUT)
# eegNet = EEGNet(nb_classes=4, Chans=PCA_COMPONENTS, Samples=640, dropoutRate=0.3, kernLength=64, F1=8, D=3, F2=16,
#                   dropoutType='Dropout')
# reshaped = pmod.SimpleFF(input_dim=[128, 64])

model = simple

print('--- Load data ---')
tf.keras.backend.clear_session()
pretraining, training = pm.load_data(subjects=range(1, 11))

if PRETRAINING:
    print('--- Pretraining ---')
    tf.keras.backend.clear_session()
    pre_hist, pre_eval, path = pm.pretraining(data=pretraining,
                                              model=model, loss=LOSS, optimizer=OPTIMIZER,
                                              batch_size=BATCH_SIZE, epochs=EPOCHS,
                                              train_size=TRAIN_SIZE,
                                              save=True, model_name=MODEL_NAME, filename=f'{MODEL_NAME}_pretrain',
                                              pca_type=PCA_TYPE, pca_components=PCA_COMPONENTS,
                                              pca_condition=PCA_CONDITION)
    pu.k_fold_visualization(pre_hist, pre_eval, batch_size=BATCH_SIZE, epochs=EPOCHS, save=True,
                            name=f"{MODEL_NAME}", folder="figures/Pretraining")
    print('--- Pretraining done ---')

if TRAINING:
    print('--- Start k-fold Cross Validation ---')
    tf.keras.backend.clear_session()
    if PRETRAINED:
        model = tf.keras.models.load_model(path)
        model.build(input_shape=(None, PCA_COMPONENTS, 640, 1))
        model.summary()
    else:
        model = model

    if PCA_TYPE == 1:
        hist, ev, _, _ = pm.ff_training(data=training,
                                        model=model, loss=LOSS, optimizer=OPTIMIZER,
                                        batch_size=BATCH_SIZE, epochs=EPOCHS_TRAIN,
                                        save=False, model_name=f'{MODEL_NAME}', filename=f'{MODEL_NAME}_Training',
                                        pca_type=PCA_TYPE, pca_components=PCA_COMPONENTS,
                                        pca_condition=PCA_CONDITION)
    else:
        hist, ev, _, _ = pm.k_fold_training(data=training,
                                            model=model, loss=LOSS, optimizer=OPTIMIZER,
                                            batch_size=BATCH_SIZE, epochs=EPOCHS_TRAIN,
                                            save=False, model_name=f'{MODEL_NAME}', filename=f'{MODEL_NAME}_Training',
                                            pca_type=PCA_TYPE, pca_components=PCA_COMPONENTS,
                                            pca_condition=PCA_CONDITION)
    pu.k_fold_visualization(hist, ev, batch_size=BATCH_SIZE, epochs=EPOCHS, save=True,
                            name=f"{MODEL_NAME}")
    print('--- k-fold Cross Validation done ---')

if INDIV_SUBJECTS:
    tf.keras.backend.clear_session()
    if PRETRAINED:
        model = tf.keras.models.load_model(path)
    else:
        model = model
    print('--- Start k-fold Cross Validation for each subject ---')
    for subj in range(1, 11):
        _, training = pm.load_data(subjects=[subj])
        print('Train Model on subj ', str(subj))
        if PCA_TYPE == 1:
            hist, ev, _, _ = pm.ff_training(data=training,
                                            model=model, loss=LOSS, optimizer=OPTIMIZER,
                                            batch_size=BATCH_SIZE, epochs=EPOCHS_TRAIN, folds=FOLDS,
                                            save=False, model_name=f'{MODEL_NAME}',
                                            filename=f'{MODEL_NAME}_Subject{subj}_Training',
                                            pca_type=PCA_TYPE, pca_components=PCA_COMPONENTS,
                                            pca_condition=PCA_CONDITION)
        else:
            hist, ev, _, _ = pm.k_fold_training(data=training,
                                                model=model, loss=LOSS, optimizer=OPTIMIZER,
                                                batch_size=BATCH_SIZE, epochs=EPOCHS_TRAIN, folds=FOLDS,
                                                save=False, model_name=f'{MODEL_NAME}',
                                                filename=f'{MODEL_NAME}_Subject{subj}_Training',
                                                pca_type=PCA_TYPE, pca_components=PCA_COMPONENTS,
                                                pca_condition=PCA_CONDITION)
        pu.k_fold_visualization(hist, ev, batch_size=BATCH_SIZE, epochs=EPOCHS, save=True,
                                name=f"{MODEL_NAME}_Subject{subj}")
    print('--- k-fold Cross Validation for each subject done ---')
