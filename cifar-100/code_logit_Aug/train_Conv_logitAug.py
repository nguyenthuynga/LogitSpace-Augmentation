from datagen_ConvNext_Sayakpaul import *
import pickle
import math
import os
"""using datagen with constant Augmentation"""

gpu=1
import gc

os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
from tensorflow.keras.callbacks import LearningRateScheduler, Callback, TensorBoard
import tensorflow as tf
import numpy as np
import tensorboard
import datetime
import albumentations as A
import matplotlib.pyplot as plt
import ConNext_Sayakpaul
import loss

"""categorical_crossentropy instead of octoberloss"""


os.environ['KMP_DUPLICATE_LIB_OK']='True'# is to advoid [SpyderKernelApp] WARNING | No such comm: 


start_lr  = 0.0002
rate_decay = 0.98

batch_size = 90
epoch_size = int(10000)
epoch_size_val = int(5000)

epochs = 100
print('rate decays ... times after all epochs : ' , rate_decay**epochs)

model_name = 'Cif100_ConvNextBase224_LogitAugLearningSigma_batchsize{}'.format(batch_size)

path_base='/home/a6000/Documents/Nga/ML_technique/cifar-100/Data/'

work_dir = path_base.replace('/Data/','/')

weights_dir = work_dir + 'weights/'
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

def load_training_history():
    try:
        f = open(weights_dir + 'info_' + model_name + '.pickle', 'rb')
        return pickle.load(f)
    except:
        return [[], [], [], [],[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

def remove_weight_acc(path,score,name):
    files=glob.glob(path+'*')
    for f in files:
        if name in f:
            s=int(f.split('_')[-1].split('.')[0])
            if score>s:
                os.remove(f)

metric_topK=loss.MetricsAtTopK(k=1)

#metric_threshold05=loss.MetricsAtThreshold(t=0.5)


metric_1=tf.keras.metrics.TopKCategoricalAccuracy(k=1)
metric_1_name='top_k_categorical_accuracy'

metric_f3=metric_topK.f3_topK
metric_f3_name='f3_topK'

metric_f1=metric_topK.f1_topK
metric_f1_name='f1_topK'

metric_recall=metric_topK.recall_topK
metric_recall_name='recall_topK'

metric_precision=metric_topK.precision_topK
metric_precision_name='precision_topK'

metric=[metric_1,metric_f1,metric_f3,metric_recall,metric_precision]


load_model=False

if load_model==False:
    epstop=0
    m = ConNext_Sayakpaul.get_model_AugBeforeLogit_learningSigma() 
    m.compile(loss= {'z':'categorical_crossentropy','z1':'categorical_crossentropy',
                'z2':'categorical_crossentropy','z3':'categorical_crossentropy'},
                optimizer='adam', metrics=metric)
else:
    epstop=0
    m=tf.keras.models.load_model(weights_dir+'save_1KDiseases_28Feb_ConNextBase384_SayakPaul_batchsize30_fold4_0_72',
     custom_objects={'octoberloss2210_c20_class_subclass': loss.octoberloss2210_c20_class_subclass,
     'octoberloss2210_c10_group': loss.octoberloss2210_c10_group, 'octoberloss2210_c5_bodypart': loss.octoberloss2210_c5_bodypart,
      'octoberloss2210_c1_other': loss.octoberloss2210_c1_other, 'f3_m':loss.f3_m})



d = load_training_history()

A1 = 0

   
for ep in range(epochs-epstop):
    tf.keras.backend.clear_session()
    gc.collect()

    print(f'\n Epoch {ep+epstop} \n')
    
    train_data = data_generator_ConvNext_Multipleoutputs(path_base,batch_size,ep+epstop, is_validation_mode=False)
    

    val_data = data_generator_ConvNext_Multipleoutputs(path_base,batch_size,ep+epstop, is_validation_mode=True)
    
    hist = m.fit(
        train_data,
        validation_data=val_data,
        steps_per_epoch=math.ceil(epoch_size / batch_size), 
        validation_steps=math.ceil(epoch_size_val / batch_size),
        verbose=1,
        epochs=1,
        callbacks=[LearningRateScheduler(lambda ep: start_lr * rate_decay ** (epstop+ep))]
     )
    
    d[0].append(hist.history['z_loss']) 
    d[1].append(hist.history['val_z_loss'])

    d[2].append(hist.history['z_'+metric_1_name])
    d[3].append(hist.history['val_z_'+metric_1_name])


    d[4].append(hist.history['z_'+metric_f3_name])
    d[5].append(hist.history['val_z_'+metric_f3_name])

    d[6].append(hist.history['z_'+metric_recall_name])
    d[7].append(hist.history['val_z_'+metric_recall_name])

    d[8].append(hist.history['z_'+metric_precision_name])
    d[9].append(hist.history['val_z_'+metric_precision_name])

    d[10].append(hist.history['z_'+metric_f1_name])
    d[11].append(hist.history['val_z_'+metric_f1_name])

    
    fig = show_figure(d)
    fig.savefig(weights_dir + model_name  + '.png')

    
    with open(weights_dir + 'info_' + model_name+ '.pickle','wb') as f:
        pickle.dump(d, f)
    
    score = int(d[3][-1][0]*1000)


    if score>A1 or ep%15==0:
        A1=score
        
        #remove_weight_acc(weights_dir,score,'model_' + model_name + '_'  )
        #m.save(weights_dir + 'model_' + model_name )
        
       