# batchnorm-adversarial-robustness

- Added .ipynb to share the code as they would have outputs as well. But found that .ipynb files are not opening in browser(maybe because of the size of outputs).
One has to clone this git project locally and fire up jupyter to view them.

- Alternately, to see just the code written in .ipynb files, I've added .py files corresponding to each of the .ipynb files.

- 4 Attacks considered here: CarliniWagner - L2 Norm (CW-L2), DeepFool, BasicIterativeMethod (BIM) and FastSignGradientMethod (FGSM)

- CIFAR
  - <b>cifar10-vanilla-and-bn-models.ipynb/.py</b>: Code For Training 4 models on CIFAR along with Adversarial Generation of 4 Attacks: 
    - CNN
    - CNN+BN
    - RESNET20
    - RESNET20+BN
  - <b>cifar-adv_training.ipynb/.py</b>: Code For Adversarial Training 4 models on CIFAR along with Evaluation using Accuracies and Loss Sensitivity:
    - CNN+Adversarial_Trained_FOR_BIM_ATTACK, 
    - CNN+BN+Adversarial_Trained_FOR_BIM_ATTACK, 
    - RESNET+Adversarial_Trained_FOR_BIM_ATTACK, 
    - RESNET+BN+Adversarial_Trained_FOR_BIM_ATTACK
  - <b>cifar10-adv-detector.ipynb/.py</b>: Ensemble ADVersarial Detector for CIFAR
- MNIST
  - <b>mnist-vanilla.ipynb/.py</b> - Code For Clean and Adversarial Training of 4 models on MNIST along with Adversarial Generation of 4 Attacks and evaluation using Accuracies and Loss Sensitivity: 
    - CNN, 
    - CNN+BN, 
    - CNN+Adversarial_Trained_FOR_BIM_ATTACK, 
    - CNN+BN+Adversarial_Trained_FOR_BIM_ATTACK
- VAE ADVERSARIAL DETECTOR APPROACH: 
  - <b>vae_encoding_and_detector.ipynb/.py</b> - VAE Based Adversarial Detector Attempted for CIFAR10
