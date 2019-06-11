(x_train, y_train, x_test, y_test) = pickle_load('/home/surthi/models/vanilla_clf_train_test_data.pkl')
(x_train_bim_adv, x_test_bim_adv) = pickle_load('/home/surthi/models/vanilla_clf_bim_xtrain_xtest.pkl')
(x_train_fgsm_adv, x_test_fgsm_adv) = pickle_load('/home/surthi/models/vanilla_clf_fgsm_data.pkl')
(x_train_deepfool_adv, x_test_deepfool_adv) = pickle_load('/home/surthi/models/vanilla_clf_deepfool_xtrain_xtest.pkl')
(x_train_cl2_adv, x_test_cl2_adv) = pickle_load('/home/surthi/models/vanilla_clf_cl2_xtrain_xtest.pkl')

# LOAD MODELS AND EVALUATE DIFFERENT COMBINATIONS OF ENSEMBLE DETECTOR which detects adversarial from clean image based on agreement between BASELINE-CNN-without-BatchNorm and ADV-TRAINED Models. If they both agree, then its considered as clean image else it considered as adversarial image 

cnn_clf_bn, model1_bn = load_clf('/home/surthi/models/cifar10/', 'vanilla_clf_with_bn.h5', 'vanilla_clf_with_bn_model.h5')
cnn_clf_adv_final, cnn_model_adv_final = load_clf('/home/surthi/models/cifar10/', 'cnn_clf_adv1_trained_final.h5', 'cnn_bn_model_adv1_trained_final.h5')

resnet = tf.keras.models.load_model('/home/surthi/adversarial-robustness-toolbox/saved_models/no_bn_cifar10_ResNet20v1_model.054.h5')
resnet_bn = tf.keras.models.load_model('/home/surthi/adversarial-robustness-toolbox/saved_models/_cifar10_ResNet20v1_model.050.h5')
resnet_clf_adv_final, resnet_model_adv_final = load_clf('/home/surthi/models/cifar10/', 'resnet_clf_adv1_trained_180_epochs.h5', 'resnet_model_adv1_trained_180_epochs.h5')

# Recording TruePositive Rates and False Positie Rates per attack type as per Carlini Wagner recommendation
def adv_detection(clf, clf_adv, x, is_x_clean):
	preds1 = np.argmax(clf.predict(x), axis=1)
	preds2 = np.argmax(clf_adv.predict(x), axis=1)
	pred_adv_detection = preds1 == preds2
	if is_x_clean:
		expected_adv_detection = np.ones(pred_adv_detection.shape, dtype=int)
	else:
	        expected_adv_detection = np.zeros(pred_adv_detection.shape, dtype=int)
	print(classification_report(expected_adv_detection, pred_adv_detection))
	

# Adv Detection with Ensemble of cnn_clf and cnn_clf_adv_final gave best results
adv_detection(cnn_clf, cnn_clf_adv_final, x_test, True)
adv_detection(cnn_clf, cnn_clf_adv_final, x_test_bim_adv, False)
adv_detection(cnn_clf, cnn_clf_adv_final, x_test_fgsm_adv, False)
adv_detection(cnn_clf, cnn_clf_adv_final, x_test_deepfool_adv, False)


# Adv Detection with Ensemble of cnn_bn and cnn_clf_adv_final
adv_detection(cnn_clf_bn, cnn_clf_adv_final, x_test, True)
adv_detection(cnn_clf_bn, cnn_clf_adv_final, x_test_bim_adv, False)
adv_detection(cnn_clf_bn, cnn_clf_adv_final, x_test_fgsm_adv, False)
adv_detection(cnn_clf_bn, cnn_clf_adv_final, x_test_deepfool_adv, False)


# Adv Detection with Ensemble of resnet and resnet_clf_adv_final
adv_detection(resnet, resnet_clf_adv_final, x_test, True)
adv_detection(resnet, resnet_clf_adv_final, x_test_bim_adv, False)
adv_detection(resnet, resnet_clf_adv_final, x_test_fgsm_adv, False)
adv_detection(resnet, resnet_clf_adv_final, x_test_deepfool_adv, False)


# Adv Detection with Ensemble of resnet_bn and resnet_clf_adv_final
adv_detection(resnet_bn, resnet_clf_adv_final, x_test, True)
adv_detection(resnet_bn, resnet_clf_adv_final, x_test_bim_adv, False)
adv_detection(resnet_bn, resnet_clf_adv_final, x_test_fgsm_adv, False)
adv_detection(resnet_bn, resnet_clf_adv_final, x_test_deepfool_adv, False)


# Testing 3 models ensembles and 2 models ensemble and recording combined (clean+adv data) accuracies and F1-Scores
from sklearn.metrics import classification_report
def f1score_3ensemble(clf_clean, clf_bn, clf_adv_trained, x_test_clean, x_test_adv):
    clean_clean_preds = np.argmax(clf_clean.predict(x_test_clean), axis=1)
    clean_bn_preds    = np.argmax(clf_bn.predict(x_test_clean), axis=1)
    clean_adv_preds   = np.argmax(clf_adv_trained.predict(x_test_clean), axis=1)

    adv_clean_preds = np.argmax(clf_clean.predict(x_test_adv), axis=1)
    adv_bn_preds    = np.argmax(clf_bn.predict(x_test_adv), axis=1)
    adv_adv_preds   = np.argmax(clf_adv_trained.predict(x_test_adv), axis=1)

    clean_is_deepfool_cwl2 = clean_clean_preds != clean_bn_preds
    clean_is_bim_fgsm = clean_clean_preds != clean_adv_preds
    clean_pred = np.logical_or(clean_is_deepfool_cwl2, clean_is_bim_fgsm)
    print(clean_is_deepfool_cwl2.shape, clean_is_bim_fgsm.shape, clean_pred.shape)

    adv_is_deepfool_cwl2 = adv_clean_preds != adv_bn_preds
    adv_is_bim_fgsm = adv_clean_preds != adv_adv_preds
    adv_pred = np.logical_or(adv_is_deepfool_cwl2, adv_is_bim_fgsm)

    clean_expected = np.zeros(clean_pred.shape, dtype=bool)
    adv_expected = np.ones(clean_pred.shape, dtype=bool)

    return classification_report(np.concatenate((clean_expected, adv_expected)), np.concatenate((clean_pred, adv_pred)))


bim_score = f1score_3ensemble(cnn_clf, resnet_bn, cnn_clf_adv_final, x_test[:3000], x_test_bim_adv[:3000])
fgsm_score = f1score_3ensemble(cnn_clf, resnet_bn, cnn_clf_adv_final, x_test[:3000], x_test_fgsm_adv[:3000])
deepfool_score = f1score_3ensemble(cnn_clf, resnet_bn, cnn_clf_adv_final, x_test[:1000], x_test_deepfool_adv[:1000])
cl2_score = f1score_3ensemble(cnn_clf, resnet_bn, cnn_clf_adv_final, x_test[:1000], x_test_cl2_adv[:1000])
print(bim_score)
print(fgsm_score)
print(cl2_score)
print(deepfool_score)
