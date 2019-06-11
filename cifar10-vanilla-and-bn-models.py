from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import tensorflow as tf
import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout, BatchNormalization
from keras.regularizers import l2
import numpy as np
from art.attacks import DeepFool
from art.classifiers import KerasClassifier
from art.utils import load_dataset
from util import *

# load data
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('cifar10')

# vanilla baseline cnn model for cifar10
def get_vanilla_model(x_train, y_train, batch_norm=False):
#     os.chdir('/home/surthi')
#     _m = tf.keras.models.load_model('data/lid_model_cifar.h5')
    m = Sequential()

    m.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
    m.add(Activation('relu'))
    if(batch_norm):
        m.add(BatchNormalization())
    m.add(Conv2D(32, (3, 3), padding='same'))
    m.add(Activation('relu'))
    if(batch_norm):
        m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    # layer 5
    m.add(Conv2D(64, (3, 3), padding='same'))
    m.add(Activation('relu'))
    if(batch_norm):
        m.add(BatchNormalization())
    m.add(Conv2D(64, (3, 3), padding='same'))
    m.add(Activation('relu'))
    if(batch_norm):
        m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    # layer 10
    m.add(Conv2D(128, (3, 3), padding='same'))
    m.add(Activation('relu'))
    if(batch_norm):
        m.add(BatchNormalization())
    m.add(Conv2D(128, (3, 3), padding='same'))
    m.add(Activation('relu'))
    if(batch_norm):
        m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))

    # layer 15
    m.add(Flatten())
    m.add(Dropout(0.5))
    m.add(Dense(1024, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
    m.add(Activation('relu'))
    if(batch_norm):
        m.add(BatchNormalization())

    # layer19
    m.add(Dropout(0.5))
    m.add(Dense(512, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))
    m.add(Activation('relu'))
    if(batch_norm):
        m.add(BatchNormalization())
    m.add(Dropout(0.5))
    m.add(Dense(10))
    m.add(Activation('softmax'))
    m.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    c = KerasClassifier(model=m)
    c.fit(x_train, y_train, nb_epochs=50, batch_size=128)
#     os.chdir('/home/surthi/adversarial-robustness-toolbox/')
    return c

# train cifar cnn with batchnorm
vanilla_clf_with_bn = get_vanilla_model(x_train, y_train, batch_norm=True)
# print training and test accuracies
evaluate(vanilla_clf_with_bn, x_train, y_train, x_test, y_test)
# save classifier
save_clf(vanilla_clf_with_bn, '/home/surthi/models/cifar10/', 'vanilla_clf_with_bn.h5', 'vanilla_clf_with_bn_model.h5')

# train cifar cnn without batchnorm, save it and print its train and test accuracies
vanilla_clf = get_vanilla_model(x_train, y_train)
save_clf(vanilla_clf, '/home/surthi/models/cifar10/', 'vanilla_clf.h5', 'vanilla_clf_model.h5')
evaluate(vanilla_clf, x_train, y_train, x_test, y_test)

# generating adversarials
def fgsm(clf, x_train, x_test, epsilon=0.1):
    from art.attacks.fast_gradient import FastGradientMethod
    epsilon = .1  # Maximum perturbation
    fgsm_adv_crafter = FastGradientMethod(clf, eps=epsilon)
    x_test_fgsm_adv = fgsm_adv_crafter.generate(x=x_test)
    x_train_fgsm_adv = fgsm_adv_crafter.generate(x=x_train)
    return x_train_fgsm_adv, x_test_fgsm_adv

def ifgsm(clf, x_train, x_test, epsilon=0.1, max_iter=10):
    from art.attacks.iterative_method import BasicIterativeMethod
    ifgsm_adv_crafter = BasicIterativeMethod(clf, eps=epsilon, eps_step=0.1, max_iter=max_iter)
    x_test_ifgsm_adv = ifgsm_adv_crafter.generate(x=x_test)
    x_train_ifgsm_adv = ifgsm_adv_crafter.generate(x=x_train)
    return x_train_ifgsm_adv, x_test_ifgsm_adv

def deepfool(clf, x_train, x_test):
    from art.attacks import DeepFool
    deep_adv_crafter = DeepFool(clf)
    x_train_deepfool_adv = deep_adv_crafter.generate(x_train)
    x_test_deepfool_adv = deep_adv_crafter.generate(x_test)
    return x_train_deepfool_adv, x_test_deepfool_adv

def carlinil2(clf, x_train, x_test):
    from art.attacks import CarliniL2Method
    cl2_adv_crafter = CarliniL2Method(classifier=clf, targeted=False, max_iter=5)
    x_train_cl2_adv = cl2_adv_crafter.generate(x_train)
    x_test_cl2_adv = cl2_adv_crafter.generate(x_test)
    return x_train_cl2_adv, x_test_cl2_adv

# (50,000, 10,000) train and test adversarials for fgsm and bim
x_train_fgsm_adv, x_test_fgsm_adv = fgsm(c1, x_train, x_test)
evaluate(vanilla_clf, x_train_fgsm_adv, y_train, x_test_fgsm_adv, y_test)
x_train_ifgsm_adv, x_test_ifgsm_adv = ifgsm(c1, x_train, x_test)
evaluate(vanilla_clf, x_train_ifgsm_adv, y_train, x_test_ifgsm_adv, y_test)

# (10,000, 2,000) train and test adversarials for deepfool and cwl2
x_train_cl2_adv, x_test_cl2_adv = carlinil2(vanilla_clf, x_train[:5000], x_test[:1000])
evaluate(vanilla_clf, x_train_cl2_adv, y_train[:5000], x_test_cl2_adv, y_test[:1000])
x_train_cl2_adv_5to10, x_test_cl2_adv_5to10 = carlinil2(vanilla_clf, x_train[5000:10000], x_test[1000:2000])
evaluate(vanilla_clf, x_train_cl2_adv_5to10, y_train[5000:10000], x_test_cl2_adv_5to10, y_test[1000:2000])

x_train_deepfool_adv, x_test_deepfool_adv = carlinil2(vanilla_clf, x_train[:5000], x_test[:1000])
evaluate(vanilla_clf, x_train_deepfool_adv, y_train[:5000], x_test_deepfool_adv, y_test[:1000])
x_train_deepfool_adv_5to10, x_test_deepfool_adv_5to10 = carlinil2(vanilla_clf, x_train[5000:10000], x_test[1000:2000])
evaluate(vanilla_clf, x_train_deepfool_adv_5to10, y_train[5000:10000], x_test_deepfool_adv_5to10, y_test[1000:2000])

# Evaluate accuracies of models against attacks
c1 = vanilla_clf
c1_bn = vanilla_clf_bn
train_acc, test_acc = evaluate(c1, x_train, y_train, x_test, y_test)
fgsm_train_acc, fgsm_test_acc = evaluate(c1, x_train_fgsm_adv, y_train, x_test_fgsm_adv, y_test)
ifgsm_train_acc, ifgsm_test_acc = evaluate(c1, x_train_ifgsm_adv, y_train, x_test_ifgsm_adv, y_test)
df_train_acc, df_test_acc = evaluate(c1, x_train_deepfool_adv, y_train[:5000], x_test_deepfool_adv, y_test[:1000])
df_train_acc_2, df_test_acc_2 = evaluate(c1, x_train_deepfool_adv_5to10, y_train[5000:10000], x_test_deepfool_adv_5to10, y_test[1000:2000])
cl2_train_acc, cl2_test_acc = evaluate(c1, x_train_cl2_adv, y_train[:5000], x_test_cl2_adv, y_test[:1000])
cl2_train_acc_2, cl2_test_acc_2 = evaluate(c1, x_train_cl2_adv_5to10, y_train[5000:10000], x_test_cl2_adv_5to10, y_test[1000:2000])

train_acc_bn, test_acc_bn = evaluate(c1_bn, x_train, y_train, x_test, y_test)
fgsm_train_acc_bn, fgsm_test_acc_bn = evaluate(c1_bn, x_train_fgsm_adv, y_train, x_test_fgsm_adv, y_test)
ifgsm_train_acc_bn, ifgsm_test_acc_bn = evaluate(c1_bn, x_train_ifgsm_adv, y_train, x_test_ifgsm_adv, y_test)
df_train_acc_bn, df_test_acc_bn = evaluate(c1_bn, x_train_deepfool_adv, y_train[:5000], x_test_deepfool_adv, y_test[:1000])
df_train_acc_2_bn, df_test_acc_2_bn = evaluate(c1_bn, x_train_deepfool_adv_5to10, y_train[5000:10000], x_test_deepfool_adv_5to10, y_test[1000:2000])
cl2_train_acc_bn, cl2_test_acc_bn = evaluate(c1_bn, x_train_cl2_adv, y_train[:5000], x_test_cl2_adv, y_test[:1000])
cl2_train_acc_2_bn, cl2_test_acc_2_bn = evaluate(c1_bn, x_train_cl2_adv_5to10, y_train[5000:10000], x_test_cl2_adv_5to10, y_test[1000:2000])

# CIFAR RESNET MODELS
def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, batch_normalization=True, num_classes=10):
    """ResNet Version 1 Model builder [a]

    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=batch_normalization)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, batch_normalization=True, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=batch_normalization)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    if batch_normalization:
	x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

def fitResNet(batch_normalization=True):
    batch_size = 32  # orig paper trained all networks with batch_size=128
    epochs = 200
    data_augmentation = True
    num_classes = 10

    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = True

    n = 3

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 1

    # Computed depth from supplied model parameter n
    if version == 1:
        depth = n * 6 + 2
     elif version == 2:
        depth = n * 9 + 2

     # Model name, depth and version
     model_type = 'ResNet%dv%d' % (depth, version)

     # Load the CIFAR10 data.
     (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Input image dimensions.
    input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth, batch_normalization=Ture)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth), batch_normalization=True

    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
    model.summary()
    print(model_type)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    _c = KerasClassifier(model=model)
    _c.fit(x_train, y_train,
          batch_size=batch_size,
          nb_epochs=50,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=callbacks)


    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    save_clf(_c, '/home/surthi/models/cifar10/', 'resnet_clf_bn.h5', 'resnet_clf_model_bn.h5')
    return _c

cifar_resnet_bn = fitResNet(batch_normalization=True)
cifar_resnet_no_bn = fitResNet(batch_normalization=False)
