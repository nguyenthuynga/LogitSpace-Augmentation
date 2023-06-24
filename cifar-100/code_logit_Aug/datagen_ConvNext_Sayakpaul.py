import glob
import pickle
import random
import numpy as np
import cv2
import albumentations as A
from albumentations.core.transforms_interface import DualTransform

import matplotlib.pyplot as plt
import os

"""remove sorting list_class_subclass,etc since they are sorted already"""


EPOCH_begin=70
EPOCH_middle=40

size=224

def getLabel(fname):
    Y=np.zeros(100)
    index=int(fname.split('/')[-1].split('.')[0].split('_')[-1])
    Y[index]=1

    return Y

         
train_transform =  A.Compose([A.LongestMaxSize(max_size=size,interpolation=cv2.INTER_LINEAR,p=1),
                              A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT,value=[0, 0, 0], p=1.0),
                             A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5), A.Rotate(limit=20,p=1.0),A.RandomRotate90(p=0.5),
                            A.Cutout(num_holes=7, max_h_size=50, max_w_size=50,  p=0.95),
                            ])
    
train_transform_middle =  A.Compose([A.LongestMaxSize(max_size=size,interpolation=cv2.INTER_LINEAR,p=1),
                              A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT,value=[0, 0, 0], p=1.0),
                             A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5), A.Rotate(limit=10,p=0.9),A.RandomRotate90(p=0.5),
                            A.Cutout(num_holes=5, max_h_size=50, max_w_size=50,  p=0.9),
                            ])
    
train_transform_final =  A.Compose([A.LongestMaxSize(max_size=size,interpolation=cv2.INTER_LINEAR,p=1),
                              A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT,value=[0, 0, 0], p=1.0),
                             A.VerticalFlip(p=0.5), A.HorizontalFlip(p=0.5), A.Rotate(limit=10,p=0.9),A.RandomRotate90(p=0.5),
                            A.Cutout(num_holes=5, max_h_size=50, max_w_size=50,  p=0.9),
                            ])

valid_transform = A.Compose([A.LongestMaxSize(max_size=size,interpolation=cv2.INTER_LINEAR,p=1),
                             A.PadIfNeeded(min_height=size, min_width=size, border_mode=cv2.BORDER_CONSTANT,value=[0, 0, 0], p=1.0),
     A.RandomCrop(size, size,  p=1.0)])

def transformer(img,epoch,val): 
    if val==False:
        if epoch<EPOCH_begin:
            transformed = train_transform(image=img)
        elif epoch<EPOCH_begin+EPOCH_middle:
            transformed = train_transform_middle(image=img)
        else:
            transformed = train_transform_final(image=img)
        img= transformed["image"]
    else:
        transformed = valid_transform(image=img)
        img = transformed["image"]
    return img

mean = [0.5071* 255, 0.4865* 255, 0.4409* 255]
std = [0.2673* 255, 0.2564* 255, 0.2762* 255]

def _preprocess(img):
    img = img 
    img = (img - mean) / std
    return img

# epoch=0

def data_generator_ConvNext(path_base,batch_size,epoch, is_validation_mode=False):
    while True: # -mean/std before processing
        data = chose_data(path_base,is_validation_mode) 

        for i in range(0, len(data), batch_size): 
            batch = data[i: i + batch_size] 
            X = []
            Y=[]
            
            for fname in batch:
                img = transformer(_preprocess((cv2.imread(fname))),epoch,val=is_validation_mode)
                Y.append(getLabel(fname))
                X.append(img)


            yield np.array(X), np.array(Y)

def data_generator_ConvNext_Multipleoutputs(path_base,batch_size,epoch, is_validation_mode=False):
    while True: # -mean/std before processing
        data = chose_data(path_base,is_validation_mode) 

        for i in range(0, len(data), batch_size): 
            batch = data[i: i + batch_size] 
            X = []
            Y=[]
            
            for fname in batch:
                img = transformer(_preprocess((cv2.imread(fname))),epoch,val=is_validation_mode)
                Y.append(getLabel(fname))
                X.append(img)


            yield np.array(X), [np.array(Y),np.array(Y),np.array(Y),np.array(Y)]
            
def chose_data(path_base,is_validation_mode):
    if is_validation_mode==False:
        train_set=['train']
    else:
        train_set=['test']
        
    choices=[]
    for folder in train_set:  

        choices+=glob.glob( path_base+folder+ "/*" )

    random.shuffle(choices)
    return choices



def show_figure(d): 
    metric_1_name='top_1_categorical_accuracy'

    metric_f3_name='f3 score'
    metric_f1_name='f1 score'

    metric_recall_name='recall'

    metric_precision_name='precision'

    fig, axes = plt.subplots(2,3)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    plt.gca().cla()
        
    axes[0][0].plot(d[0],label='train loss' )
    axes[0][0].plot(d[1],label='validation loss')
    axes[0][0].legend()
    axes[0][0].set_xlabel('loss')
    axes[0][0].grid(True)

    axes[0][1].plot(d[2],label='train '+  metric_1_name)
    axes[0][1].plot(d[3],label='validation ' + metric_1_name)
    axes[0][1].legend()
    axes[0][1].set_xlabel(metric_1_name)
    axes[0][1].grid(True)

    axes[0][2].plot(d[10],label='train '+  metric_f1_name)
    axes[0][2].plot(d[11],label='validation ' + metric_f1_name)
    axes[0][2].legend()
    axes[0][2].set_xlabel(metric_f1_name)
    axes[0][2].grid(True)

    axes[1][0].plot(d[4],label='train '+  metric_f3_name)
    axes[1][0].plot(d[5],label='validation ' + metric_f3_name)
    axes[1][0].legend()
    axes[1][0].set_xlabel(metric_f3_name)
    axes[1][0].grid(True)

    axes[1][1].plot(d[6],label='train '+  metric_recall_name)
    axes[1][1].plot(d[7],label='validation ' + metric_recall_name)
    axes[1][1].legend()
    axes[1][1].set_xlabel(metric_recall_name)
    axes[1][1].grid(True)

    axes[1][2].plot(d[8],label='train '+  metric_precision_name)
    axes[1][2].plot(d[9],label='validation ' + metric_precision_name)
    axes[1][2].legend()
    axes[1][2].set_xlabel(metric_precision_name)
    axes[1][2].grid(True)


    plt.draw()
    plt.show(block = False)

    return fig

    
