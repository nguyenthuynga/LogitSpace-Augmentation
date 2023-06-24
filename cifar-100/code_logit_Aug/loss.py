from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy#,mean_squared_error
import tensorflow.keras.backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import binary_accuracy

class MetricsAtTopK:
    def __init__(self, k):
        self.k = k

    def _get_prediction_tensor(self, y_pred):
        """Takes y_pred and creates a tensor of same shape with 1 in indices where, the values are in top_k
        """
        topk_values, topk_indices = tf.nn.top_k(y_pred, k=self.k, sorted=False, name="topk")
        # the topk_indices are along last axis (1). Add indices for axis=0
        ii, _ = tf.meshgrid(tf.range(tf.shape(y_pred)[0]), tf.range(self.k), indexing='ij')
        index_tensor = tf.reshape(tf.stack([ii, topk_indices], axis=-1), shape=(-1, 2))
        prediction_tensor = tf.compat.v1.sparse_to_dense(sparse_indices=index_tensor,
                                               output_shape=tf.shape(y_pred),
                                               default_value=0,
                                               sparse_values=1.0,
                                               validate_indices=False
                                               )
        prediction_tensor = tf.cast(prediction_tensor, K.floatx())
        return prediction_tensor

    def true_positives_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        return true_positive

    def false_positives_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c2 = K.sum(prediction_tensor)  # TP + FP
        false_positive = c2 - true_positive
        return false_positive

    def false_negatives_at_k(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c3 = K.sum(y_true)  # TP + FN
        false_negative = c3 - true_positive
        return false_negative

    def precision_topK(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c2 = K.sum(prediction_tensor)  # TP + FP
        return true_positive/(c2+K.epsilon())

    def recall_topK(self, y_true, y_pred):
        prediction_tensor = self._get_prediction_tensor(y_pred=y_pred)
        true_positive = K.sum(tf.multiply(prediction_tensor, y_true))
        c3 = K.sum(y_true)  # TP + FN
        return true_positive/(c3+K.epsilon())

    def f1_topK(self, y_true, y_pred):
        precision = self.precision_topK(y_true=y_true, y_pred=y_pred)
        recall = self.recall_topK(y_true=y_true, y_pred=y_pred)
        f1 = (2*precision*recall)/(precision+recall+K.epsilon())
        return f1
    def f3_topK(self, y_true, y_pred):
        score=3
        precision = self.precision_topK(y_true=y_true, y_pred=y_pred)
        recall = self.recall_topK(y_true=y_true, y_pred=y_pred)
        return (1+score**2)*((precision*recall)/((score**2*precision+recall+K.epsilon())))


"""for f1 socre , code taken from here https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model"""
class MetricsAtThreshold:
    def __init__(self, t):
        self.threshold_value = t

    
    def recall_threshold(self,y_true, y_pred):
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), self.threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # Compute the number of positive targets.
        possible_positives = K.sum(K.clip(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + K.epsilon())
        return recall_ratio


    def precision_threshold(self,y_true, y_pred):
        # Adaptation of the "round()" used before to get the predictions. Clipping to make sure that the predicted raw values are between 0 and 1.
        y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), self.threshold_value), K.floatx())
        # Compute the number of true positives. Rounding in prevention to make sure we have an integer.
        true_positives = K.round(K.sum(K.clip(y_true * y_pred, 0, 1)))
        # count the predicted positives
        predicted_positives = K.sum(y_pred)
        # Get the precision ratio
        precision_ratio = true_positives / (predicted_positives + K.epsilon())
        return precision_ratio

    def f3_threshold(self,y_true, y_pred): #using this
        score=3
        precision = self.precision_threshold(y_true, y_pred)
        recall = self.recall_threshold(y_true, y_pred)
        return (1+score**2)*((precision*recall)/((score**2*precision+recall+K.epsilon())))

    def f1_threshold(self,y_true, y_pred): #using this
        precision = self.precision_threshold(y_true, y_pred)
        recall = self.recall_threshold(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

def octoberloss2210_c5_bodypart(y_true,y_pred): #using this
    c=5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1=(1-y_true_f)*K.maximum(y_pred_f-0.2,0)+c*y_true_f*K.maximum(0.8-y_pred_f,0)
    return 0.2*l1
def octoberloss2210_c5_bodypart_c_increasingto05(y_true,y_pred): #using this
    c=5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1=(1-y_true_f)*K.maximum(y_pred_f-0.2,0)+c*y_true_f*K.maximum(0.8-y_pred_f,0)
    return 0.5*l1

def octoberloss2210_c10_group(y_true,y_pred): #using this
    c=10
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1=(1-y_true_f)*K.maximum(y_pred_f-0.2,0)+c*y_true_f*K.maximum(0.8-y_pred_f,0)
    return 0.5*l1

def octoberloss2210_c20_class_subclass(y_true,y_pred): #using this
    c=20
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1=(1-y_true_f)*K.maximum(y_pred_f-0.2,0)+c*y_true_f*K.maximum(0.8-y_pred_f,0)
    return 1*l1



def octoberloss2210_c1_other(y_true,y_pred): #using this
    c=5
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1=(1-y_true_f)*K.maximum(y_pred_f-0.2,0)+c*y_true_f*K.maximum(0.8-y_pred_f,0)
    return 0.2*l1

def octoberloss0208_c60(y_true,y_pred):
    c=60
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1=(1-y_true_f)*K.maximum(y_pred_f-0.2,0)+c*y_true_f*K.maximum(0.8-y_pred_f,0)
    return l1
def octoberloss0208_vector_c60(y_true,y_pred):
    c=60
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1=(1-y_true_f)*K.maximum(y_pred_f-0.2,0)+c*y_true_f*K.maximum(0.8-y_pred_f,0)
    return K.mean(l1)

def binary_crossentropy01(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)/10

def mean_squared_error_W1(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def cube_loss(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true)**3, axis=-1)

def sensitive_new025(y_true, y_pred):
    smooth=0.0000001
    y_true_=K.cast(y_true>0.00,dtype='float32')
#    y_true_=K.maximum(y_true,0)
    x=y_true_*(K.cast(y_pred>0.25,dtype='float32'))
    return K.sum(x)/(K.sum(y_true_)+smooth)

def sensitive_new00(y_true, y_pred):
    smooth=0.0000001
    y_true_=K.cast(y_true>0.0,dtype='float32')
#    y_true_=K.maximum(y_true,0)
    x=y_true_*(K.cast(y_pred>0.0,dtype='float32'))
    return K.sum(x)/(K.sum(y_true_)+smooth)

def specific_new025(y_true, y_pred):
    smooth=0.0000001
    y_true_=K.cast(y_true<0.0,dtype='float32')
#    y_true_=0-K.minimum(y_true,0)
    x=y_true_*(K.cast(y_pred<-0.25,dtype='float32'))
    return K.sum(x)/(K.sum(y_true_)+smooth)

def specific_new00(y_true, y_pred):
    smooth=0.0000001
    y_true_=K.cast(y_true<0.0,dtype='float32')
#    y_true_=0-K.minimum(y_true,0)
    x=y_true_*(K.cast(y_pred<0.0,dtype='float32'))
    return K.sum(x)/(K.sum(y_true_)+smooth)

def nosloss4(y_true,y_pred):
    c=1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1= K.maximum(-y_pred_f*0.5,0)+(K.maximum(K.minimum(0.5 - y_pred_f,0.5),0))**4
    l2= K.maximum(y_pred_f*0.5,0)+(K.maximum(K.minimum(0.5 + y_pred_f,0.5),0))**4
    return 10*K.mean(c*(y_true_f+1)*l1+(1-y_true_f)*l2)

def nosloss2(y_true,y_pred):
    c=4
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1= K.maximum(-y_pred_f,0)+(K.maximum(K.minimum(0.5 - y_pred_f,0.5),0))**2#y>0
    l2= K.maximum(y_pred_f,0)+(K.maximum(K.minimum(0.5 + y_pred_f,0.5),0))**2#y<0
    return 10*K.mean(c*(y_true_f+1)*l1+(1-y_true_f)*l2)

def _huber_loss( y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond  = K.abs(error) <= clip_delta

    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

    return K.mean(tf.where(cond, squared_loss, quadratic_loss))
def _huber_loss_matrix( y_true, y_pred, clip_delta=1.0):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    
    error = y_true - y_pred
    cond  = K.abs(error) <= clip_delta

    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


def octoberloss_square0208(y_true,y_pred):
    c=8
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1= -K.minimum(y_pred_f,0.5)+0.5+(K.maximum(K.minimum(0.8 - y_pred_f,0.3),0))**2#ytrue>0.5
    l2= K.maximum(y_pred_f,0.5)-0.5+(K.maximum(K.minimum(-0.2 + y_pred_f,0.3),0))**2#ytrue<0.5
    return K.mean(c*y_true_f*l1+(1-y_true_f)*l2)

def octoberloss_square0109(y_true,y_pred):
    c=8
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1= -K.minimum(y_pred_f,0.5)+0.5+(K.maximum(K.minimum(0.9 - y_pred_f,0.4),0))**2#ytrue>0.5
    l2= K.maximum(y_pred_f,0.5)-0.5+(K.maximum(K.minimum(-0.1 + y_pred_f,0.4),0))**2#ytrue<0.5
    return K.mean(c*y_true_f*l1+(1-y_true_f)*l2)

def l1no(y_pred_f):
    return np.maximum(-y_pred_f,0)+(np.maximum(np.minimum(0.5 - y_pred_f,0.5),0))**2#y>0
def l2no(y_pred_f):
    return np.maximum(y_pred_f,0)+(np.maximum(np.minimum(0.5 + y_pred_f,0.5),0))**2#y<0

def octoberloss_0208(y_true,y_pred):
    c=8
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
#    l1=(1-y_true_f)*K.maximum(y_pred_f-0.2,0)+c*y_true_f*K.maximum(0.8-y_pred_f,0)
    return K.mean((1-y_true_f)*K.maximum(y_pred_f-0.2,0)+c*y_true_f*K.maximum(0.8-y_pred_f,0))
def octoberloss_015085(y_true,y_pred):
    c=6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    l1=(1-y_true_f)*K.maximum(y_pred_f-0.15,0)+c*y_true_f*K.maximum(0.85-y_pred_f,0)
    return l1
def abinaryloss6mean(y_true,y_pred):
    k=6
    c=8
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    f1= y_true_f*((y_true_f-y_pred_f)**k)
    f2= (1- y_true_f)*((y_true_f-y_pred_f)**k)
    return K.mean(8*(c*f1+f2)*(2**k)/k)

def sensitive30(y_true, y_pred):
    smooth=0.0000001
    x=y_true*(K.cast(y_pred>0.30,dtype='float32'))
    return K.sum(x)/(K.sum(y_true)+smooth)
def specific70(y_true, y_pred):
    smooth=0.0000001
    x=(1-y_true)*(K.cast(y_pred<0.7,dtype='float32'))
    return K.sum(x)/(K.sum(1-y_true)+smooth)
def sensitive50(y_true, y_pred):
    smooth=0.0000001
    x=y_true*(K.cast(y_pred>0.50,dtype='float32'))
    return K.sum(x)/(K.sum(y_true)+smooth)
def specific50(y_true, y_pred):
    smooth=0.0000001
    x=(1-y_true)*(K.cast(y_pred<0.5,dtype='float32'))
    return K.sum(x)/(K.sum(1-y_true)+smooth)

def sensitive15(y_true, y_pred):
    smooth=0.0000001
    x=y_true*(K.cast(y_pred>0.15,dtype='float32'))
    return K.sum(x)/(K.sum(y_true)+smooth)
def specific85(y_true, y_pred):
    smooth=0.0000001
    x=(1-y_true)*(K.cast(y_pred<0.85,dtype='float32'))
    return K.sum(x)/(K.sum(1-y_true)+smooth)

def sensitive10(y_true, y_pred):
    smooth=0.0000001
    x=y_true*(K.cast(y_pred>0.10,dtype='float32'))
    return K.sum(x)/(K.sum(y_true)+smooth)
def specific90(y_true, y_pred):
    smooth=0.0000001
    x=(1-y_true)*(K.cast(y_pred<0.9,dtype='float32'))
    return K.sum(x)/(K.sum(1-y_true)+smooth)



def loss_similarity(y_true, y_pred):
    c=6
    d=3
    y_pred=0-tf.math.top_k(0-K.abs(y_pred),k=30)[0]
    y= K.sum(y_pred,axis=-1)
    y = K.minimum(y,5)*0.2
    return c*(1-y_true)*(y_true-y)**2+d*y_true*(y_true-y)**2

def acc_similarity(y_true, y_pred):
    y_pred=0-tf.math.top_k(0-K.abs(y_pred),k=30)[0]
    y= K.sum(y_pred,axis=-1)
    y = K.minimum(y,5)*0.2
    return binary_accuracy(y_true, y)

def binary(y_true, y_pred):
    return binary_accuracy(K.sum(K.abs(y_pred),axis=-1)/2, y_pred/2)
    
def abinaryloss24_c02(y_true,y_pred):
    c=0.2
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    f1= y_true_f*((y_true_f-y_pred_f)**2 + (y_true_f-y_pred_f)**4)
    f2= (1- y_true_f)*((y_true_f-y_pred_f)**2 + (y_true_f-y_pred_f)**4)
    return (c*K.sum(f1)+K.sum(f2))*5

def abinaryloss8(y_true,y_pred):
    c=2
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    f1= y_true_f*((y_true_f-y_pred_f)**8)
    f2= (1- y_true_f)*((y_true_f-y_pred_f)**8)
    return (c*f1+f2)*100

def abinaryloss6(y_true,y_pred):
    k=6
    c=8
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    f1= y_true_f*((y_true_f-y_pred_f)**k)
    f2= (1- y_true_f)*((y_true_f-y_pred_f)**k)
    return 8*(c*f1+f2)*(2**k)/k

def jaccard(y_true, y_pred):
    smooth=0.001
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(K.round(y_pred))
    intersection = K.sum(y_true_f * y_pred_f,axis=-1)
    jac=(smooth+intersection)/(smooth-intersection+K.sum(y_pred_f,axis=-1)+K.sum(y_true_f, axis=-1))
    return jac

def size_error(y_true,y_pred):
    smooth= 0.00001
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(K.round(y_pred))
    error= abs(K.sum(y_pred_f-y_true_f,axis=-1))/(K.sum(y_true_f,axis=-1)+smooth)
    return error

def score(y_true, y_pred):
    s=(10-K.relu(10-K.relu(K.round(20*jaccard(y_true, y_pred)-9.5))))/10
    return K.mean(s)

def t_classloss(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    k=2.5
    f= K.abs(y_true_f-y_pred_f)**k
    return K.mean(f)*10 

def t_classloss24(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    f= (y_true_f-y_pred_f)**2 + (y_true_f-y_pred_f)**4
    return K.mean(f)*5

def t_scoreb(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y1 = y_pred_f*y_true_f
    y2 = (1- y_pred_f)*(1-y_true_f)
    k=2.5
    f1= 1-(1-y1)**k
    f2= (1-y2)**k
    score = K.sum(f1)/K.sum(f2)
    return score 

def t_lossb(y_true,y_pred):    
    return 1- t_scoreb(y_true,y_pred)

def t_score2(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y1 = y_pred_f*y_true_f
    y2 = (1- y_pred_f)*(1-y_true_f)
    k=2
    f1= 1-(1-y1)**k
    f2= (1-y2)**k
    score = K.sum(f1)/K.sum(f2)
    return score 

def t_loss2(y_true,y_pred):    
    return 1- t_scoreb(y_true,y_pred)

def newaccuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.sign(y_pred)), axis=-1)

def accuracy_class_test0(y_true, y_pred):
    i=0
    yt=K.flatten(y_true[i,:])
    return tf.shape(yt)#9
def accuracy_classtest1(y_true, y_pred):
    i=1
    yt=K.flatten(y_true[:,i])
    yp=K.flatten(y_pred[:,i])
    return tf.shape(yt)#4

def categorical_accuracy(y_true, y_pred):
    return K.cast(K.equal(K.argmax(y_true, axis=-1),
                          K.argmax(y_pred, axis=-1)),
                  K.floatx())


def dice_coeffi(y_true, y_pred):
    smooth=10**-6
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = 2 * K.sum(y_true_f * y_pred_f, axis=1)
#    union = K.sum(y_true_f * y_true_f, axis=1) + K.sum(y_pred_f * y_pred_f, axis=1)
    union = K.sum(y_true_f, axis=1) + K.sum(y_pred_f, axis=1)
    return K.mean((smooth+intersection) /(smooth+ union))

def dice_lossi(y_true, y_pred):
    loss = 1 - dice_coeffi(y_true, y_pred)
    return loss


def bce_dice_lossi(y_true, y_pred):
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)

    bce=K.mean(binary_crossentropy(y_true_f, y_pred_f))
    dice=dice_lossi(y_true, y_pred)
    return bce+dice

def dice_coeffb(y_true, y_pred):
    smooth=10**-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = 2 * K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)
    return (smooth+intersection) /(smooth+ union)

def dice_lossb(y_true, y_pred):
    loss = 1 - dice_coeffb(y_true, y_pred)
    return loss

def bce_dice_lossb_binary(y_true, y_pred):
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)
    bce=binary_crossentropy(y_true_f, y_pred_f)
    dice=dice_lossb(y_true, y_pred)
    return bce+dice
def bce_dice_lossb_categorical(y_true, y_pred):
    y_true_f = K.clip(K.batch_flatten(y_true), K.epsilon(), 1.)
    y_pred_f = K.clip(K.batch_flatten(y_pred), K.epsilon(), 1.)
    bce=categorical_crossentropy(y_true_f, y_pred_f)
    dice=dice_lossb(y_true, y_pred)
    return bce+dice
