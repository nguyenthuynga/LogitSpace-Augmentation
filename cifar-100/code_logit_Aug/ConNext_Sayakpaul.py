
from tensorflow import keras
import tensorflow_hub as hub
import os
#os.environ["CUDA_VISIBLE_DEVICES"]=str(0)


MODEL_PATH = "https://tfhub.dev/sayakpaul/convnext_tiny_1k_224_fe/1"
MODEL_PATH_small= "https://tfhub.dev/sayakpaul/convnext_small_1k_224_fe/1"
MODEL_PATH_base= "https://tfhub.dev/sayakpaul/convnext_base_1k_224_fe/1"

def get_model0(model_path=MODEL_PATH_base, res=224, num_classes=100):
    hub_layer = hub.KerasLayer(model_path, trainable=True)

    model = keras.Sequential(
        [
            keras.layers.InputLayer((res, res, 3)),
            hub_layer,
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model

#tensorflow.compat.v1.disable_eager_execution()
import tensorflow


def get_model_aug_logit(model_path=MODEL_PATH_base, res=224, num_classes=100,sigma=0.2):
    input = keras.layers.Input((res, res, 3))
    x = hub.KerasLayer(model_path, trainable=True)(input)

    z = keras.layers.Dense(num_classes)(x)

    #batch_size=tensorflow.compat.v1.placeholder(tensorflow.int32, shape=[])
    #noise_shape = tensorflow.stack([batch_size, num_classes])

    noise_shape = tensorflow.stack([90,num_classes])#if take 1D like this, will same noise for all images in same batch!!!!!! Need to fix this!!!!!!
    #print(tensorflow.shape(z))

    ep1=tensorflow.random.uniform(noise_shape,-sigma,sigma)
    z1=keras.layers.Multiply()([z, ep1])

    ep2=tensorflow.random.uniform(noise_shape,-sigma,sigma)
    z2=keras.layers.Multiply()([z, ep2])

    ep3=tensorflow.random.uniform(noise_shape,-sigma,sigma)
    z3=keras.layers.Multiply()([z, ep3])

    out1=keras.layers.Add()([z1,z])
    out2=keras.layers.Add()([z2,z])
    out3=keras.layers.Add()([z3,z])

    z=keras.layers.Activation(keras.activations.softmax, name='z')(z)
    out1=keras.layers.Activation(keras.activations.softmax, name='z1')(out1)
    out2=keras.layers.Activation(keras.activations.softmax, name='z2')(out2)
    out3=keras.layers.Activation(keras.activations.softmax, name='z3')(out3)


    outputs=[z,out1,out2,out3]
    model = keras.Model(input, outputs)
    print(model.summary())
    return model



def get_model_aug_SigmoidbeforeLogit_FixedSigma(model_path=MODEL_PATH_base, res=224, num_classes=100,sigma=0.4):
    input = keras.layers.Input((res, res, 3))
    x = hub.KerasLayer(model_path, trainable=True)(input)

    z = keras.layers.Dense(num_classes,activation='sigmoid',name='z')(x)

    #batch_size=tensorflow.compat.v1.placeholder(tensorflow.int32, shape=[])
    #noise_shape = tensorflow.stack([batch_size, num_classes])

    noise_shape = tensorflow.stack([90,num_classes])#if take 1D like this, will same noise for all images in same batch!!!!!! Need to fix this!!!!!!
    #print(tensorflow.shape(z))

    ep1=tensorflow.random.uniform(noise_shape,-sigma,sigma)
    z1=keras.layers.Multiply()([z, ep1])

    ep2=tensorflow.random.uniform(noise_shape,-sigma,sigma)
    z2=keras.layers.Multiply()([z, ep2])

    ep3=tensorflow.random.uniform(noise_shape,-sigma,sigma)
    z3=keras.layers.Multiply()([z, ep3])

    out1=keras.layers.Add(name='z1')([z1,z])
    out2=keras.layers.Add(name='z2')([z2,z])
    out3=keras.layers.Add(name='z3')([z3,z])

    outputs=[z,out1,out2,out3]
    model = keras.Model(input, outputs)
    print(model.summary())
    return model

def get_model_AugBeforeLogit_learningSigma(model_path=MODEL_PATH_base, res=224, num_classes=100):
    input = keras.layers.Input((res, res, 3))
    x = hub.KerasLayer(model_path, trainable=True)(input)
    z = keras.layers.Dense(num_classes)(x)
    sigma = keras.layers.Dense(num_classes)(x)

    noise_shape = tensorflow.stack([90,num_classes])#if take 1D like this, will same noise for all images in same batch!!!!!! Need to fix this!!!!!!

    ep1=tensorflow.random.uniform(noise_shape,-1,1)
    sigma1=keras.layers.Multiply()([sigma, ep1])

    ep2=tensorflow.random.uniform(noise_shape,-1,1)
    sigma2=keras.layers.Multiply()([sigma, ep2])

    ep3=tensorflow.random.uniform(noise_shape,-1,1)
    sigma3=keras.layers.Multiply()([sigma, ep3])

    out1=keras.layers.Add()([sigma1,z])
    out2=keras.layers.Add()([sigma2,z])
    out3=keras.layers.Add()([sigma3,z])

    z=keras.layers.Activation(keras.activations.softmax, name='z')(z)
    out1=keras.layers.Activation(keras.activations.softmax, name='z1')(out1)
    out2=keras.layers.Activation(keras.activations.softmax, name='z2')(out2)
    out3=keras.layers.Activation(keras.activations.softmax, name='z3')(out3)

    outputs=[z,out1,out2,out3]
    model = keras.Model(input, outputs)
    print(model.summary())
    return model

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

def get_inception_resnet_classificationCommonSkin_1Output(classes=100,input_shape=(224,224,3)):
    base_model = InceptionResNetV2(input_shape=input_shape,include_top=False)
    x= base_model.get_layer('conv_7b').output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.25)(x)
    x = keras.layers.Dense(classes, activation ='sigmoid', name='all_features')(x)
    
    model = keras.Model(base_model.input, x)
    print(model.summary())
    return model