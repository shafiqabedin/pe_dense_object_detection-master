from itertools import product

from functools import partial
from keras import backend as K
from keras.losses import categorical_crossentropy


def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth / 2) / (K.sum(y_true,
                                                                axis=axis) + K.sum(y_pred,
                                                                                   axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def multislice_dice_coef(y_true, y_pred, smooth=0.):
    y_true = y_true[..., 4]
    y_true = K.batch_flatten(y_true)
    y_true = K.cast(y_true, dtype='int32')
    y_true = K.one_hot(y_true, 2)

    y_true = y_true[..., 1]

    y_pred = y_pred[..., 1]
    y_pred = K.batch_flatten(y_pred)

    intersection = K.sum(y_true * y_pred, 1) + smooth
    denom = K.sum(y_true ** 2, 1) + K.sum(y_pred ** 2, 1) + smooth
    dice = (2. * intersection) / denom

    return K.mean(dice)


def multislice_dice_loss(y_true, y_pred):
    return 1. - multislice_dice_coef(y_true, y_pred)


def sensitivity_specificity_loss(y_true, y_pred, r=0.05):
    y_true = y_true[..., 4]
    y_true = K.batch_flatten(y_true)
    y_true = K.cast(y_true, dtype='int32')
    y_true = K.one_hot(y_true, 2)

    y_true = y_true[..., 1]

    y_pred = y_pred[..., 1]
    y_pred = K.batch_flatten(y_pred)

    y_true_one_cold = 1. - y_true
    squared_error = K.tf.square(y_true - y_pred)
    epsilon_denominator = 1e-5

    specificity_part = K.tf.reduce_sum(squared_error * y_true, 0) / (K.tf.reduce_sum(y_true, 0) + epsilon_denominator)

    sensitivity_part = (K.tf.reduce_sum(K.tf.multiply(squared_error, y_true_one_cold), 0) / (
            K.tf.reduce_sum(y_true_one_cold, 0) + epsilon_denominator))

    return K.tf.reduce_sum(r * specificity_part + (1 - r) * sensitivity_part)


def weighted_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


def weighted_pixelwise_crossentropy(y_true, y_pred, class_weights):
    # epsilon = _to_tensor(_EPSILON, y_pred.dtype.base_dtype)
    epsilon = K.epsilon()
    y_pred = K.tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    return - K.tf.reduce_sum(K.tf.multiply(y_true * K.tf.log(y_pred), class_weights))


def categorical_focal_loss(y_true, y_pred, gamma=2., alpha=.25):
    # Scale predictions so that the class probs of each sample sum to 1
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

    # Calculate Cross Entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Calculate Focal Loss
    loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

    # Sum the losses in mini_batch
    return K.sum(loss, axis=1)


def multislice_categorical_focal_loss(y_true, y_pred):
    y_true = y_true[..., 4]
    y_true = K.cast(y_true, dtype='int32')
    y_true = K.one_hot(y_true, 2)

    return categorical_focal_loss(y_true, y_pred)


def ignore_unknown_xentropy(ytrue, ypred):
    return (1 - ytrue[:, :, 0]) * categorical_crossentropy(ytrue, ypred)


def continuous_dice_loss(y_true, y_pred):
    y_true = y_true[..., 4]
    y_true = K.batch_flatten(y_true)
    y_true = K.cast(y_true, dtype='int32')
    y_true = K.one_hot(y_true, 2)

    y_true = y_true[..., 1]
    y_pred = y_pred[..., 1]

    intersection = K.sum(y_true * y_pred, 1)
    denom = K.sum(y_true ** 2, 1) + K.sum(y_pred ** 2, 1)
    dice = 1. - (2. * intersection) / denom
    return K.mean(dice)


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
