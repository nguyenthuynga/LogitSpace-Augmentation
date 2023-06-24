from datagen_ConvNext_Sayakpaul import *
import pickle
import math
import os
#os.nice(0)

gpu=3
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



os.environ['KMP_DUPLICATE_LIB_OK']='True'# is to advoid [SpyderKernelApp] WARNING | No such comm: 


start_lr  = 0.0002
rate_decay = 0.986

batch_size = 100 
epoch_size = int(10000)
epoch_size_val = int(5000)

epochs = 250

model_name = 'Cifar10_ConvNextTiny224_ConstantAug{}'.format(batch_size)

path_base='/home/a100//Documents/Nga/Augmentation_paper/cifar-10/Data/'

work_dir = path_base.replace('/Data/','/')

weights_dir = work_dir + 'weights/'
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

def load_training_history():
    try:
        f = open(weights_dir + 'info_' + model_name + '.pickle', 'rb')
        return pickle.load(f)
    except:
        return [[], [], [], [],]

def remove_weight_acc(path,score,name):
    files=glob.glob(path+'*')
    for f in files:
        if name in f:
            s=int(f.split('_')[-1].split('.')[0])
            if score>s:
                os.remove(f)

                

metric_f3=tf.keras.metrics.TopKCategoricalAccuracy(k=1)
metric_f3_name='top_k_categorical_accuracy'

metric=[metric_f3]

load_model=False

if load_model==False:
    epstop=0
    m = ConNext_Sayakpaul.get_model0() 
    m.compile(loss=loss.octoberloss2210_c1,
                optimizer='adam', metrics=metric_f3)
else:
    epstop=0
    m=tf.keras.models.load_model(weights_dir+'save_1KDiseases_28Feb_ConNextBase384_SayakPaul_batchsize30_fold4_0_72',
     custom_objects={'octoberloss2210_c20_class_subclass': loss.octoberloss2210_c20_class_subclass,
     'octoberloss2210_c10_group': loss.octoberloss2210_c10_group, 'octoberloss2210_c5_bodypart': loss.octoberloss2210_c5_bodypart,
      'octoberloss2210_c1_other': loss.octoberloss2210_c1_other, 'f3_m':loss.f3_m})



d = load_training_history()

A1 = 0

   
for ep in range(epochs):
    tf.keras.backend.clear_session()
    gc.collect()

    print(f'\n Epoch {ep} \n')
    
    train_data = data_generator_ConvNext(path_base,batch_size,ep+epstop, is_validation_mode=False)
    

    val_data = data_generator_ConvNext(path_base,batch_size*2,ep+epstop, is_validation_mode=True)
    
    hist = m.fit(
        train_data,
        validation_data=val_data,
        steps_per_epoch=math.ceil(epoch_size / batch_size), 
        validation_steps=math.ceil(epoch_size_val / batch_size/2),
        epochs=1,
        callbacks=[LearningRateScheduler(lambda ep: start_lr * rate_decay ** (epstop+ep))]
     )
    
    d[0].append(hist.history['loss']) 

    d[1].append(hist.history[metric_f3_name])

    d[2].append(hist.history['val_loss'])

    d[3].append(hist.history['val_'+metric_f3_name])
    
    fig = show_figure(d)
    fig.savefig(weights_dir + model_name  + '.png')
    
    with open(weights_dir + 'info_' + model_name+ '.pickle','wb') as f:
        pickle.dump(d, f)
    
    score = int(d[3][-1][0]*1000)


    if score>A1 or ep%15==0:
        A1=score
        
        #remove_weight_acc(weights_dir,score,'model_' + model_name + '_'  )
        m.save(weights_dir + 'model_' + model_name )
        
       
