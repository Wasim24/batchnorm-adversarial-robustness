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
from art.data_generators import KerasDataGenerator
from art.defences import AdversarialTrainer
from util import *

# load data and models trained in cifar10-vanilla-and-bn-models
(g_x_train, g_y_train, g_x_test, g_y_test) = pickle_load('/home/surthi/models/vanilla_clf_train_test_data.pkl')
(g_x_train_bim_adv, g_x_test_bim_adv) = pickle_load('/home/surthi/models/vanilla_clf_bim_xtrain_xtest.pkl')

cnn_clf_bn, cnn_model_bn = load_clf('/home/surthi/models/cifar10/', 'vanilla_clf_with_bn.h5', 'vanilla_clf_with_bn_model.h5')
cnn_clf, cnn_model = load_clf('/home/surthi/models/cifar10/', 'vanilla_clf.h5', 'vanilla_clf_model.h5')

resnet_clf, resnet_model = load_clf('/home/surthi/models/cifar10/', 'resnet_clf.h5', 'resnet_clf_model.h5')
resnet_clf_bn, resnet_model_bn = load_clf('/home/surthi/models/cifar10/', 'resnet_clf_bn.h5', 'resnet_clf_model_bn.h5')

# Adversarial Training with 1 epoch of clean data and 1 epoch of BIM adversarial data
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
	resnet_clf_bn_adv_trained_1 = adv_training_1(resnet_clf_bn, g_x_train, g_y_train, g_x_train_bim_adv, g_x_test_bim_adv, g_x_test, g_y_test, epochs=5)
	resnet_clf_adv_trained_1 = adv_training_1(resnet_clf, g_x_train, g_y_train, g_x_train_bim_adv, g_x_test_bim_adv, g_x_test, g_y_test, epochs=5)
	cnn_clf_bn_adv_trained = adv_training_1(cnn_clf_bn, g_x_train, g_y_train, g_x_train_bim_adv, g_x_test_bim_adv, g_x_test, g_y_test, epochs=5)
	cnn_clf_adv_trained = adv_training_1(cnn_clf, g_x_train, g_y_train, g_x_train_bim_adv, g_x_test_bim_adv, g_x_test, g_y_test, epochs=5)

save_clf(cnn_clf_bn_adv_trained_1, '/home/surthi/models/cifar10/', 'cnn_bn_clf_adv1_trained_final_again1.h5', 'cnn_bn_model_adv1_trained_final_again1.h5')
save_clf(cnn_clf_adv_trained_1, '/home/surthi/models/cifar10/', 'cnn_clf_adv1_trained_final_again1.h5', 'cnn_model_adv1_trained_final_again1.h5')
save_clf(resnet_clf_bn_adv_trained_1, '/home/surthi/models/cifar10/', 'resnet_bn_clf_adv1_trained_final_again1.h5', 'resnet_bn_model_adv1_trained_final_again1.h5')
save_clf(resnet_clf_adv_trained_1, '/home/surthi/models/cifar10/', 'resnet_clf_adv1_trained_final_again1.h5', 'resnet_model_adv1_trained_final_again1.h5')

# Evaluating Accuracies on CLEAN AND 4 ATTACK ADVERSARIALS
evaluate(cnn_clf_adv_final, g_x_train, g_y_train, g_x_test, g_y_test)
evaluate(cnn_clf_adv_final, g_x_train_bim_adv, g_y_train, g_x_test_bim_adv, g_y_test)
evaluate(resnet_clf_adv_final, g_x_train, g_y_train, g_x_test, g_y_test)
evaluate(resnet_clf_adv_final, g_x_train_bim_adv, g_y_train, g_x_test_bim_adv, g_y_test)

# Compute Loss Sensitivity for all the models
import numpy.linalg as la
def loss_sensitivity(classifier, x, y):
    grads = classifier.loss_gradient(x, y)
    norm = la.norm(grads.reshape(grads.shape[0], -1), ord=2, axis=1)
    return np.mean(norm)

c1 = cnn_clf
c1_bn = cnn_clf_bn
ls_clean_bn = loss_sensitivity(c1_bn, x_train[:5000], y_train[:5000])
ls_clean = loss_sensitivity(c1, x_train[:5000], y_train[:5000])

ls_df_bn = loss_sensitivity(c1_bn, x_train_deepfool_adv_5to10, y_train[5000:10000])
ls_df_bn_2 = loss_sensitivity(c1_bn, x_train_deepfool_adv, y_train[:5000])

ls_cl2_bn = loss_sensitivity(c1_bn, x_train_cl2_adv_5to10, y_train[5000:10000])
ls_cl2_bn_2 = loss_sensitivity(c1_bn, x_train_cl2_adv, y_train[:5000])

ls_fgsm_bn = loss_sensitivity(c1_bn, x_train_fgsm_adv[:5000], y_train[:5000])
ls_ifgsm_bn = loss_sensitivity(c1_bn, x_train_ifgsm_adv[:5000], y_train[:5000])

ls_df = loss_sensitivity(c1, x_train_deepfool_adv_5to10, y_train[5000:10000])
ls_df_2 = loss_sensitivity(c1, x_train_deepfool_adv, y_train[:5000])

ls_cl2 = loss_sensitivity(c1, x_train_cl2_adv_5to10, y_train[5000:10000])
ls_cl2_2 = loss_sensitivity(c1, x_train_cl2_adv, y_train[:5000])

ls_fgsm = loss_sensitivity(c1, x_train_fgsm_adv[:5000], y_train[:5000])
ls_ifgsm = loss_sensitivity(c1, x_train_ifgsm_adv[:5000], y_train[:5000])

adv_ls_clean_bn = loss_sensitivity(cnn_clf_adv_final, x_train[:3000], y_train[:3000])
adv_ls_clean = loss_sensitivity(cnn_clf_adv_final, x_train[:3000], y_train[:3000])

adv_ls_df_bn = loss_sensitivity(cnn_bn_clf_adv_final, x_train_deepfool_adv, y_train[:3000])
adv_ls_cl2_bn = loss_sensitivity(cnn_bn_clf_adv_final, x_train_cl2_adv, y_train[:3000])
adv_ls_fgsm_bn = loss_sensitivity(cnn_bn_clf_adv_final, x_train_fgsm_adv[:3000], y_train[:3000])
adv_ls_ifgsm_bn = loss_sensitivity(cnn_bn_clf_adv_final, x_train_bim_adv[:3000], y_train[:3000])

adv_ls_df = loss_sensitivity(cnn_clf_adv_final, x_train_deepfool_adv, y_train[:3000])
adv_ls_cl2 = loss_sensitivity(cnn_clf_adv_final, x_train_cl2_adv, y_train[:3000])
adv_ls_fgsm = loss_sensitivity(cnn_clf_adv_final, x_train_fgsm_adv[:3000], y_train[:3000])
adv_ls_ifgsm = loss_sensitivity(cnn_clf_adv_final, x_train_bim_adv[:3000], y_train[:3000])

rn_adv_ls_clean_bn = loss_sensitivity(resnet_clf_adv_final, x_train[:3000], y_train[:3000])
rn_adv_ls_clean = loss_sensitivity(resnet_clf_adv_final, x_train[:3000], y_train[:3000])

rn_adv_ls_df_bn = loss_sensitivity(resnet_bn_clf_adv_final, x_train_deepfool_adv, y_train[:3000])
rn_adv_ls_cl2_bn = loss_sensitivity(resnet_bn_clf_adv_final, x_train_cl2_adv, y_train[:3000])
rn_adv_ls_fgsm_bn = loss_sensitivity(resnet_bn_clf_adv_final, x_train_fgsm_adv[:3000], y_train[:3000])
rn_adv_ls_ifgsm_bn = loss_sensitivity(resnet_bn_clf_adv_final, x_train_bim_adv[:3000], y_train[:3000])

rn_adv_ls_df = loss_sensitivity(resnet_clf_adv_final, x_train_deepfool_adv, y_train[:3000])
rn_adv_ls_cl2 = loss_sensitivity(resnet_clf_adv_final, x_train_cl2_adv, y_train[:3000])
rn_adv_ls_fgsm = loss_sensitivity(resnet_clf_adv_final, x_train_fgsm_adv[:3000], y_train[:3000])
rn_adv_ls_ifgsm = loss_sensitivity(resnet_clf_adv_final, x_train_bim_adv[:3000], y_train[:3000])

# Plot Loss Sensitivity of 8 models (CNN and RESNET), (with and without BatchNorm) (With and Without Adversarial Training)
adv_ls_bn = (adv_ls_clean_bn, adv_ls_df_bn, adv_ls_cl2_bn, adv_ls_fgsm_bn, adv_ls_ifgsm_bn)
adv_ls = (adv_ls_clean, adv_ls_df, adv_ls_cl2, adv_ls_fgsm, adv_ls_ifgsm)

rn_adv_ls_bn = (rn_adv_ls_clean_bn, rn_adv_ls_df_bn, rn_adv_ls_cl2_bn, rn_adv_ls_fgsm_bn, rn_adv_ls_ifgsm_bn)
rn_adv_ls = (rn_adv_ls_clean, rn_adv_ls_df, rn_adv_ls_cl2, rn_adv_ls_fgsm, rn_adv_ls_ifgsm)
# print(adv_ls_bn, adv_ls, rn_adv_ls_bn, rn_adv_ls)

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


ax3 = plt.subplot(2, 2, 3)
rects3 = plt.bar(index, resnet_ls, bar_width, alpha=opacity, color='b', label='Clean training')
rects4 = plt.bar(index + bar_width, rn_adv_ls, bar_width, alpha=opacity, color='g', label='Adversarial training')
plt.xlabel('Attack Types')
plt.ylabel('Loss Sensitivity')
ax3.set_title('ResNet20 Without BatchNorm')
plt.xticks(index + bar_width, ('No Attack', 'DF', 'CWL2', 'FGSM', 'BIM'))
plt.legend()

ax4 = plt.subplot(2, 2, 4)
rects7 = plt.bar(index, resnet_ls_bn, bar_width, alpha=opacity, color='b', label='Clean training')
rects8 = plt.bar(index + bar_width, rn_adv_ls_bn, bar_width, alpha=opacity, color='g', label='Adversarial training')
plt.xlabel('Attack Types')
ax4.set_title('ResNet20 With BatchNorm')
plt.xticks(index + bar_width, ('No Attack', 'DF', 'CWL2', 'FGSM', 'BIM'))
plt.legend()

# plt.constrained_layout()
plt.tight_layout()
plt.savefig('cifar10-loss-sensitivity.png')

# Compute Average Perturbation
import numpy.linalg as la
N = 3000
bim_perturbation = np.mean(la.norm((x_train-x_train_bim_adv).reshape(x_train.shape[0], -1), 2, axis=1))
fgsm_perturbation = np.mean(la.norm((x_train[:N]-x_train_fgsm_adv).reshape(N, -1), 2, axis=1))
cl2_perturbation = np.mean(la.norm((x_train[:N]-x_train_cl2_adv).reshape(N, -1), 2, axis=1))
df_perturbation = np.mean(la.norm((x_train[:N]-x_train_deepfool_adv).reshape(N, -1), 2, axis=1))

# plot perturbation
avg_perturbations = (df_perturbation, cl2_perturbation, fgsm_perturbation, bim_perturbation)

plt.figure(figsize=(4,2))
y_pos = np.arange(len(avg_perturbations))
plt.bar(y_pos, avg_perturbations, align='center', alpha=0.5)
plt.xticks(y_pos, ['DeepFool', 'CWL2', 'FGSM', 'BIM'])
plt.title('L2 Norm of Perturbations for different attacks')
# plt.xlabel('Attack Types')
plt.ylabel('Perturbation')
plt.show()
fig.savefig('cifar10-avg-perturbation.png')


