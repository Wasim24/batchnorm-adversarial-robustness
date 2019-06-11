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
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset('mnist')

# mnist cnn model
def get_vanilla_model(x_train, y_train, batch_norm=False):
    m = Sequential()

    m.add(Conv2D(64, (3, 3), padding='valid', input_shape=(28, 28, 1)))
    m.add(Activation('relu'))
    if(batch_norm):
        m.add(BatchNormalization())
    m.add(Conv2D(64, (3, 3)))
    m.add(Activation('relu'))
    if(batch_norm):
        m.add(BatchNormalization())
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.5))
    m.add(Flatten())
    m.add(Dense(128))         
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
    return c

# mnist cnn with bn training, save model and print train/test accuracies
vanilla_clf_with_bn = get_vanilla_model(x_train, y_train, batch_norm=True)
save_clf(vanilla_clf_with_bn, '/home/surthi/models/mnist/', 'vanilla_clf_with_bn.h5', 'vanilla_clf_with_bn_model.h5')
evaluate(vanilla_clf_with_bn, x_train, y_train, x_test, y_test)

# mnist cnn without bn trianing, save model and print train/test accuracies
vanilla_clf = get_vanilla_model(x_train, y_train)
save_clf(vanilla_clf, '/home/surthi/models/mnist/', 'vanilla_clf.h5', 'vanilla_clf_model.h5')
evaluate(vanilla_clf, x_train, y_train, x_test, y_test)

# Adversarial Generation

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

def fgsm(clf, x_train, x_test, epsilon=0.1):
    from art.attacks.fast_gradient import FastGradientMethod
    epsilon = .1  # Maximum perturbation
#     fgsm_adv_crafter = FastGradientMethod(clf, eps=epsilon)
#     fgsm_adv_crafter = FastGradientMethod(clf, eps=0.4, batch_size=2)
    fgsm_adv_crafter = FastGradientMethod(clf, eps=0.4, eps_step=0.01, batch_size=2)
    x_test_fgsm_adv = fgsm_adv_crafter.generate(x=x_test)
    x_train_fgsm_adv = fgsm_adv_crafter.generate(x=x_train)
    return x_train_fgsm_adv, x_test_fgsm_adv

def ifgsm(clf, x_train, x_test, epsilon=0.1, max_iter=10):
    from art.attacks.iterative_method import BasicIterativeMethod
    ifgsm_adv_crafter = BasicIterativeMethod(clf, eps=epsilon, eps_step=0.1, max_iter=max_iter)
    x_test_ifgsm_adv = ifgsm_adv_crafter.generate(x=x_test)
    x_train_ifgsm_adv = ifgsm_adv_crafter.generate(x=x_train)
    return x_train_ifgsm_adv, x_test_ifgsm_adv

x_train_fgsm_adv, x_test_fgsm_adv = fgsm(vanilla_clf, x_train, x_test, epsilon=1.0)
evaluate(vanilla_clf, x_train_fgsm_adv, y_train, x_test_fgsm_adv, y_test)
pickle_dump((x_train_fgsm_adv, x_test_fgsm_adv), 'mnist-fgsm-data.pkl')

x_train_ifgsm_adv, x_test_ifgsm_adv = ifgsm(vanilla_clf, x_train, x_test)
evaluate(vanilla_clf, x_train_ifgsm_adv, y_train, x_test_ifgsm_adv, y_test)
pickle_dump((x_train_ifgsm_adv, x_test_ifgsm_adv), 'mnist-ifgsm-data.pkl')

x_train_cl2_adv, x_test_cl2_adv = carlinil2(vanilla_clf, x_train[:5000], x_test[:1000])
evaluate(vanilla_clf, x_train_cl2_adv, y_train[:5000], x_test_cl2_adv, y_test[:1000])
pickle_dump((x_train_cl2_adv, x_test_cl2_adv), 'mnist-cl2-data.pkl')

x_train_deepfool_adv, x_test_deepfool_adv = deepfool(vanilla_clf, x_train[:5000], x_test[:1000])
evaluate(vanilla_clf, x_train_deepfool_adv, y_train[:5000], x_test_deepfool_adv, y_test[:1000])
pickle_dump((x_train_deepfool_adv, x_test_deepfool_adv), 'mnist-deepfool-data.pkl')

# Evaluating Acc of Models against adversarials
train_acc, test_acc = evaluate(vanilla_clf, x_train, y_train, x_test, y_test)
fgsm_train_acc, fgsm_test_acc = evaluate(vanilla_clf, x_train_fgsm_adv, y_train, x_test_fgsm_adv, y_test)
ifgsm_train_acc, ifgsm_test_acc = evaluate(vanilla_clf, x_train_ifgsm_adv1, y_train, x_test_ifgsm_adv1, y_test)
df_train_acc, df_test_acc = evaluate(vanilla_clf, x_train_deepfool_adv, y_train[:5000], x_test_deepfool_adv, y_test[:1000])
cl2_train_acc, cl2_test_acc = evaluate(vanilla_clf, x_train_cl2_adv, y_train[:5000], x_test_cl2_adv, y_test[:1000])

train_acc, test_acc = evaluate(vanilla_clf_with_bn, x_train, y_train, x_test, y_test)
fgsm_train_acc, fgsm_test_acc = evaluate(vanilla_clf_with_bn, x_train_fgsm_adv, y_train, x_test_fgsm_adv, y_test)
ifgsm_train_acc, ifgsm_test_acc = evaluate(vanilla_clf_with_bn, x_train_ifgsm_adv1, y_train, x_test_ifgsm_adv1, y_test)
df_train_acc, df_test_acc = evaluate(vanilla_clf_with_bn, x_train_deepfool_adv, y_train[:5000], x_test_deepfool_adv, y_test[:1000])
cl2_train_acc, cl2_test_acc = evaluate(vanilla_clf_with_bn, x_train_cl2_adv, y_train[:5000], x_test_cl2_adv, y_test[:1000])

# Adversarial Training
def adv_training_1(clf, x_train, y_train, x_train_adv, x_test_adv, x_test, y_test, epochs=5, batch_size=128):
    print("Before training:")
    evaluate(clf, x_train, y_train, x_test, y_test)    
    evaluate(clf, x_train_adv, y_train, x_test_adv, y_test)
    
    for i in range(epochs):
        clf.fit(x_train_adv, y_train, nb_epochs=1, batch_size=128)
        clf.fit(x_train, y_train, nb_epochs=1, batch_size=128)
    
    print("After training:")
    evaluate(clf, x_train, y_train, x_test, y_test)    
    evaluate(clf, x_train_adv, y_train, x_test_adv, y_test)
    return clf

for i in range(35):
    cnn_clf_bn_adv_trained = adv_training_1(vanilla_clf_bn, g_x_train, g_y_train, g_x_train_bim_adv, g_x_test_bim_adv, g_x_test, g_y_test, epochs=5)
    cnn_clf_adv_trained = adv_training_1(vanilla_clf, g_x_train, g_y_train, g_x_train_bim_adv, g_x_test_bim_adv, g_x_test, g_y_test, epochs=5)

save_clf(cnn_clf_bn_adv_trained, '/home/surthi/models/mnist/', 'cnn_clf_bn_adv_trained.h5', 'cnn_clf_bn_model_adv_trained.h5')
save_clf(cnn_clf_adv_trained, '/home/surthi/models/mnist/', 'cnn_clf_adv_trained.h5', 'cnn_clf_model_adv_trained.h5')

# Evaluating Accuracies on CLEAN AND 4 ATTACK ADVERSARIALS
evaluate(cnn_clf_adv_final, g_x_train, g_y_train, g_x_test, g_y_test)
evaluate(cnn_clf_adv_final, g_x_tcrain_bim_adv, g_y_train, g_x_test_bim_adv, g_y_test)
evaluate(cnn_clf_adv_final, g_x_train_fgsm_adv, g_y_train, g_x_test_fgsm_adv, g_y_test)
evaluate(cnn_clf_adv_final, g_x_train_cl2_adv, g_y_train, g_x_test_cl2_adv, g_y_test)
evaluate(cnn_clf_adv_final, g_x_train_deepfool_adv, g_y_train, g_x_test_deepfool_adv, g_y_test)

evaluate(cnn_clf_bn_adv_final, g_x_train, g_y_train, g_x_test, g_y_test)
evaluate(cnn_clf_bn_adv_final, g_x_train_bim_adv, g_y_train, g_x_test_bim_adv, g_y_test)
evaluate(cnn_clf_bn_adv_final, g_x_train_fgsm_adv, g_y_train, g_x_test_fgsm_adv, g_y_test)
evaluate(cnn_clf_bn_adv_final, g_x_train_cl2_adv, g_y_train, g_x_test_cl2_adv, g_y_test)
evaluate(cnn_clf_bn_adv_final, g_x_train_deepfool_adv, g_y_train, g_x_test_deepfool_adv, g_y_test)

# Loss Sensitivity
import numpy.linalg as la
def loss_sensitivity(classifier, x, y):
    grads = classifier.loss_gradient(x, y)
    norm = la.norm(grads.reshape(grads.shape[0], -1), ord=2, axis=1)
    return np.mean(norm)

ls_clean_bn = loss_sensitivity(vanilla_clf_with_bn, x_train[:3000], y_train[:3000])
ls_df_bn = loss_sensitivity(vanilla_clf_with_bn, x_train_deepfool_adv[:3000], y_train[:3000])
ls_cl2_bn = loss_sensitivity(vanilla_clf_with_bn, x_train_cl2_adv[:3000], y_train[:3000])
ls_fgsm_bn = loss_sensitivity(vanilla_clf_with_bn, x_train_fgsm_adv[:3000], y_train[:3000])
ls_ifgsm_bn = loss_sensitivity(vanilla_clf_with_bn, x_train_ifgsm_adv1[:3000], y_train[:3000])

ls_clean = loss_sensitivity(vanilla_clf, x_train[:3000], y_train[:3000])
ls_df = loss_sensitivity(vanilla_clf, x_train_deepfool_adv[:3000], y_train[:3000])
ls_cl2 = loss_sensitivity(vanilla_clf, x_train_cl2_adv[:3000], y_train[:3000])
ls_fgsm = loss_sensitivity(vanilla_clf, x_train_fgsm_adv[:3000], y_train[:3000])
ls_ifgsm = loss_sensitivity(vanilla_clf, x_train_ifgsm_adv1[:3000], y_train[:3000])

adv_ls_clean_bn = loss_sensitivity(loaded_vanilla_clf_with_bn_adv, x_train[:3000], y_train[:3000])
adv_ls_df_bn = loss_sensitivity(loaded_vanilla_clf_with_bn_adv, x_train_deepfool_adv[:3000], y_train[:3000])
adv_ls_cl2_bn = loss_sensitivity(loaded_vanilla_clf_with_bn_adv, x_train_cl2_adv[:3000], y_train[:3000])
adv_ls_fgsm_bn = loss_sensitivity(loaded_vanilla_clf_with_bn_adv, x_train_fgsm_adv[:3000], y_train[:3000])
adv_ls_ifgsm_bn = loss_sensitivity(loaded_vanilla_clf_with_bn_adv, x_train_ifgsm_adv1[:3000], y_train[:3000])

adv_ls_clean = loss_sensitivity(loaded_vanilla_clf_adv, x_train[:3000], y_train[:3000])
adv_ls_df = loss_sensitivity(loaded_vanilla_clf_adv, x_train_deepfool_adv[:3000], y_train[:3000])
adv_ls_cl2 = loss_sensitivity(loaded_vanilla_clf_adv, x_train_cl2_adv[:3000], y_train[:3000])
adv_ls_fgsm = loss_sensitivity(loaded_vanilla_clf_adv, x_train_fgsm_adv[:3000], y_train[:3000])
adv_ls_ifgsm = loss_sensitivity(loaded_vanilla_clf_adv, x_train_ifgsm_adv1[:3000], y_train[:3000])

adv_ls_bn = (adv_ls_clean_bn, adv_ls_df_bn, adv_ls_cl2_bn, adv_ls_fgsm_bn, adv_ls_ifgsm_bn)
adv_ls = (adv_ls_clean, adv_ls_df, adv_ls_cl2, adv_ls_fgsm, adv_ls_ifgsm)
ls_bn = (ls_clean_bn, ls_df_bn, ls_cl2_bn, ls_fgsm_bn, ls_ifgsm_bn)
ls = (ls_clean, ls_df, ls_cl2, ls_fgsm, ls_ifgsm)

# plot Loss Sensitivity for 4 models (cnn, cnn_bn, cnn_adv, cnn_bn_adv)
# data to plot
n_groups = 5
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
plt.figure(figsize=(8,5))
# plt.suptitle("Loss Sensitivity for Cifar10 models with and without Adversarial training")

ax1 = plt.subplot(2, 2, 1)
rects1 = plt.bar(index, ls, bar_width, alpha=opacity, color='b', label='Clean training')
rects2 = plt.bar(index + bar_width, adv_ls, bar_width, alpha=opacity, color='g', label='Adversarial training')
plt.ylabel('Loss Sensitivity')
ax1.set_title('CNN Without BatchNorm')
plt.xticks(index + bar_width, ('No Attack', 'DF', 'CWL2', 'FGSM', 'BIM'))
plt.legend()

ax2 = plt.subplot(2, 2, 2)
rects5 = plt.bar(index, ls_bn, bar_width, alpha=opacity, color='b', label='Clean training')
rects6 = plt.bar(index + bar_width, adv_ls_bn, bar_width, alpha=opacity, color='g', label='Adversarial training')
ax2.set_title('CNN With BatchNorm')
plt.xticks(index + bar_width, ('No Attack', 'DF', 'CWL2', 'FGSM', 'BIM'))
plt.legend()
