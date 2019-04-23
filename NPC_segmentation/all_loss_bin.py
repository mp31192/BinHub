from keras import backend as K
from all_index_bin import *
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import tensorflow as tf



seg_thre = 0.5

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

def EuclideanLoss(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    loss = K.sum(K.square(y_true_f - y_pred_f))
    return loss

def EuclideanLossWithWeight(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y = K.abs(y_true_f - y_pred_f)
    all_one = K.ones_like(y_true_f)
    y1_1 = K.clip(y - 0.15 * all_one, -1, 0)
    y1_sign = K.clip(-1 * K.sign(y1_1), 0, 1)
    y1 = y1_sign * y

    y2_1 = K.clip(y - 0.15 * all_one, 0, 5)
    y2_2 = K.clip(y2_1 - 0.5 * all_one, -1, 0)
    y2_sign = K.clip(-1 * K.sign(y2_2), 0, 1)
    y2 = y2_sign * y

    y3_1 = K.clip(y - 0.5 * all_one, 0, 5)
    y3_2 = K.clip(y3_1 - 0.8 * all_one, -1, 0)
    y3_sign = K.clip(-1 * K.sign(y3_2), 0, 1)
    y3 = y3_sign * y

    y4_1 = K.sign(y - 0.8 * all_one)
    y4_sign = K.clip(y4_1, 0, 1)
    y4 = y4_sign * y

    y_final = 0.6*y1 + 1 * y2 + 1.2 * y3 + 1.4 * y4

    loss = K.sum(K.square(y_final))

    return loss

def EuclideanLossminu(y_true, y_pred):
    loss = K.sum(K.abs(y_true - y_pred))
    return loss

def DiceCoefLoss(y_true,y_pred):
    return 1-DiceCoef(y_true, y_pred)

def RecallLossWeighted(y_true,y_pred):
    # weighted = 1-DiceCoefLoss(y_true,y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    recall_loss = intersection/(K.sum(y_true_f)+1)
    weighted_recall_loss = recall_loss#*weighted
    return weighted_recall_loss

def DiceLossWithRecall(y_true,y_pred):
    recall_loss = RecallLossWeighted(y_true,y_pred)
    dsc_loss = DiceCoefLoss(y_true,y_pred)
    loss = recall_loss+dsc_loss
    return loss

def DiceCoefLoss_SNIP(y_true,y_pred):
    SNIP_MASK = y_true[:,:,:,1]
    SNIP_MASK = K.expand_dims(SNIP_MASK,axis=3)
    y_true_scores = y_true[:,:,:,0]
    y_true_scores = K.expand_dims(y_true_scores,axis=3)
    y_true_SNIP = y_true_scores * SNIP_MASK
    y_pred_SNIP = y_pred * SNIP_MASK
    return  (1-DiceCoef(y_true_SNIP,y_pred_SNIP))

def CE_DICE(y_true,y_pred):
    ce_loss = weighted_binary_crossentropy(y_true,y_pred)
    ce_loss = K.mean(ce_loss)
    dsc_loss = DiceCoefLoss(y_true,y_pred)
    loss = ce_loss-K.log(1-dsc_loss)
    return loss

def focal_loss(y_true,y_pred):
    gamma = 0.75
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true,1),y_pred,tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1,1e-3,.999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha*K.pow(1. - pt_1,gamma)*K.log(pt_1))-K.sum((1. - alpha)*K.pow(pt_0,gamma)*K.log(1. - pt_0))

def focal_dice_loss(y_true,y_pred,alpha=1e-5):
    return alpha*focal_loss(y_true,y_pred) + DiceCoefLoss(y_true,y_pred)

def h1(y_true,y_pred,alpha = 1e-5):
    return alpha*focal_loss(y_true,y_pred)

def h2(y_true,y_pred):
    return DiceCoefLoss(y_true,y_pred)

def weighted_binary_crossentropy_SNIP(y_true,y_pred):
    SNIP_MASK = y_true[:,:,:,1]
    SNIP_MASK = K.expand_dims(SNIP_MASK,axis=3)
    y_true_scores = y_true[:,:,:,0]
    y_true_scores = K.expand_dims(y_true_scores,axis=3)
    y_true_SNIP = y_true_scores * SNIP_MASK
    y_pred_SNIP = y_pred * SNIP_MASK
    bc = weighted_binary_crossentropy(y_true_SNIP,y_pred_SNIP,from_logits=False)
    return  bc

def weighted_binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor with the same shape as `output`.
        output: A tensor.
        from_logits: Whether `output` is expected to be a logits tensor.
            By default, we consider that `output`
            encodes a probability distribution.

    # Returns
        A tensor.
    """
    # Note: tf.nn.weighted_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # transform back to logits
        _epsilon = _to_tensor(1e-07, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.log(output / (1 - output))

    return tf.nn.weighted_cross_entropy_with_logits(target,output,pos_weight=1)