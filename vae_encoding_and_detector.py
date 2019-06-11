import os
from __future__ import absolute_import, division, print_function, unicode_literals
!pip install tensorflow-gpu==2.0.0-alpha0
!pip uninstall tfp-nightly
!pip install tensorflow-probability
!pip install --upgrade tf-nightly-2.0-preview tfp-nightly

import collections
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import tensorflow as tf

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model, Model
tf.__version__

(X_clean_train, Y_clean_train), (X_clean_test, Y_clean_test) = tf.keras.datasets.cifar10.load_data()


input_shape = (32,32,3)
encoded_size = 16
base_depth = 32
L2_weight_decay = 2e-5
batch_size = 1000

tfk = tf.keras
tfkl = tf.keras.layers
import tensorflow_probability as tfp
tfd = tfp.distributions
initializer = tf.initializers.VarianceScaling(scale=2.0)

def SampleFromEncoding(mean, sigma):
    t_sigma = tf.sqrt(tf.exp(sigma))
    epsilon = tf.keras.backend.random_normal(shape=tf.shape(mean), mean=0., stddev=1., dtype = tf.float64)
    return mean + t_sigma * epsilon

class VaeConvNet(tf.keras.Model):
    def __init__(self, input_shape = (32,32,3)):
        super(VaeConvNet, self).__init__()  
        self.e_conv1 = tf.keras.layers.Conv2D(filters=3 , kernel_size=2, strides=1, activation=tf.nn.relu, padding='same', name="e-conv1", trainable=True)
        self.e_conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=2, activation=tf.nn.relu, padding='same', name="e-conv2", trainable=True)
        self.e_conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=1, activation=tf.nn.relu, padding='same', name="e-conv3", trainable=True)
        self.e_conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=2, strides=1, activation=tf.nn.relu, padding='same', name="e-conv4", trainable=True)
        self.e_dense  = tf.keras.layers.Dense(128, tf.nn.relu, name="e-sdnse", trainable=True)
        self.e_mean  = tf.keras.layers.Dense(encoded_size, tf.nn.softplus, name="e-mean", trainable=True)
        self.e_sigma  = tf.keras.layers.Dense(encoded_size,tf.nn.softplus, name="e-sigma", trainable=True)
        
        self.d_dense = tf.keras.layers.Dense(128, name="d-dense", trainable=True)
        self.d_dense2 = tf.keras.layers.Dense(32*input_shape[0]*input_shape[1]/4, name="d-dense2", trainable=True)
        self.d_convT1 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=1, activation=tf.nn.relu, padding='same', name="d-convT1", trainable=True)
        self.d_convT2 = tf.keras.layers.Conv2DTranspose(filters=32 , kernel_size=2, strides=1, activation=tf.nn.relu, padding='same', name="d-convT2", trainable=True)
        self.d_convT3 = tf.keras.layers.Conv2DTranspose(filters=3  , kernel_size=2, strides=2, activation=tf.nn.relu, padding='valid', name="d-convT3", trainable=True)
        #self.d_conv = tf.keras.layers.Conv2D(filters=3  , kernel_size=2, strides=1, activation=tf.nn.sigmoid, padding='same', name="d-conv", trainable=True)
        
        
    def call(self, x, training=False):
        x = self.e_conv1(x)
        x = self.e_conv2(x)
        x = self.e_conv3(x)
        x = self.e_conv4(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.e_dense(x)
        mean = self.e_mean(x)
        sigma = self.e_sigma(x)
        sample = SampleFromEncoding( mean, sigma)
        y = self.d_dense(sample)
        y = self.d_dense2(y)
        y = tf.keras.layers.Reshape(target_shape=(16,16,32), input_shape=(None, encoded_size))(y)
        y = self.d_convT1(y)
        y = self.d_convT2(y)
        y = self.d_convT3(y)
        #y = self.d_conv(y)
        return y, mean, sigma    

def VaeLoss(x, x_reco, mean, sigma):
  reconstruction_term = -tf.reduce_sum(tfp.distributions.MultivariateNormalDiag(
      tf.keras.layers.Flatten()(x_reco), scale_identity_multiplier=0.05).log_prob(tf.keras.layers.Flatten()(x)))
  kl_divergence = 0.5 * tf.reduce_mean(tf.square(mean) + tf.square(sigma) - tf.math.log(1e-8 + tf.square(sigma)) - 1, [1])
  return tf.reduce_mean(reconstruction_term + kl_divergence)

def optimizer_init_fn():
    learning_rate = 0.001
    return tf.keras.optimizers.Adam(learning_rate=learning_rate)

optimizer = optimizer_init_fn()


model     = VaeConvNet()

def Train(model, optimizer, x, num_epochs, batch_size, is_training=False):
    t = 0
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    num_batches = tf.shape(x)[0]//batch_size
    for epoch in range(num_epochs):
        
        train_loss.reset_states()
        
        for batch in range(num_batches):
            x_batch_input = x[batch*batch_size : (batch+1)*batch_size]/255
            
            with tf.GradientTape() as tape:
                # Use the model function to build the forward pass.
                y_output, mean, sigma  = model(x_batch_input, training=True)
                loss = VaeLoss(x_batch_input, y_output, mean, sigma )
                
                if is_training:
                  gradients = tape.gradient(loss, model.trainable_variables)
                  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
            # Update the metrics
            train_loss.update_state(loss)

            if t % 10 == 0:
              template = 'Iteration {}, Epoch {}, Loss: {}'
              print (template.format(t, epoch+1, train_loss.result()) )                   
            t += 1
    
# train model
Train(model, optimizer=optimizer, x=X_clean_train, num_epochs=25, batch_size = batch_size, is_training=True)
Train(model, optimizer=optimizer, x=X_clean_train, num_epochs=10, batch_size = batch_size, is_training=True)
Train(model, optimizer=optimizer, x=X_clean_train, num_epochs=10, batch_size = batch_size, is_training=True)
Train(model, optimizer=optimizer, x=X_clean_train, num_epochs=10, batch_size = batch_size, is_training=True)
Train(model, optimizer=optimizer, x=X_clean_train, num_epochs=10, batch_size = batch_size, is_training=True)
Train(model, optimizer=optimizer, x=X_clean_train, num_epochs=15, batch_size = batch_size, is_training=True)

# get mean and std dev of encoding
x_reco, mean, sigma  = model(X_clean_test[0:batch_size]/255, training=False)

# plt clean image and reconstructed image from trained vae
plt.imshow(X_clean_test[13])
plt.imshow(np.clip(x_reco[13]*255, 0, 255).astype(int))

# save model
model.save_weights('vae.h5')    
model_file = drive.CreateFile({'title' : 'vae.h5'})
model_file.SetContentFile('vae.h5')
model_file.Upload()


# VAE DETECTOR 
class VaeDetectorNet(tf.keras.Model):
    def __init__(self, input_shape = (32,32,3)):
        super(VaeDecodervNet, self).__init__() 
        self.dense1 = tf.keras.layers.Dense(256, name="d-dense1", activation=tf.nn.leaky_relu,kernel_initializer=initializer)
        self.dense2 = tf.keras.layers.Dense(64, name="d-dense2", activation=tf.nn.leaky_relu,kernel_initializer=initializer)
        self.sigmoid = tf.keras.layers.Dense(1, name="d-sigmoid", activation='sigmoid',kernel_initializer=tf.initializers.GlorotNormal())
        
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.sigmoid(x) 

# CNN based detector model
def encode_model_init_fn():
    input_shape = (32, 32, 3)
    channel_1, channel_2, channel_3, channel_4, num_classes = 32, 64, 96, 128, 10
    BN_decay        = 0.99
    BN_epsilon      = 1.e-5
    L2_weight_decay = 2e-5
    droput_rate     = 0.11
    
    initializer = tf.initializers.VarianceScaling(scale=2.0)
    layers = [
        tf.keras.layers.Conv2D(input_shape=input_shape, kernel_size=(3,3), filters=channel_1, activation='relu', padding='same', kernel_initializer=initializer, 
                             kernel_regularizer=tf.keras.regularizers.l2(L2_weight_decay),
                             bias_regularizer=tf.keras.regularizers.l2(L2_weight_decay)),
        tf.keras.layers.BatchNormalization(axis=-1, momentum=BN_decay, epsilon=BN_epsilon),
        #tf.keras.layers.Dropout(droput_rate),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=channel_2, activation='relu', padding='same', kernel_initializer=initializer,
                             kernel_regularizer=tf.keras.regularizers.l2(L2_weight_decay),
                             bias_regularizer=tf.keras.regularizers.l2(L2_weight_decay)),
        tf.keras.layers.BatchNormalization(axis=-1, momentum=BN_decay, epsilon=BN_epsilon),
        tf.keras.layers.Dropout(droput_rate),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=channel_3, activation='relu', padding='same', kernel_initializer=initializer,
                             kernel_regularizer=tf.keras.regularizers.l2(L2_weight_decay),
                             bias_regularizer=tf.keras.regularizers.l2(L2_weight_decay)),
        tf.keras.layers.BatchNormalization(axis=-1, momentum=BN_decay, epsilon=BN_epsilon),
        tf.keras.layers.Dropout(droput_rate),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=channel_4, activation='relu', padding='same', kernel_initializer=initializer,
                             kernel_regularizer=tf.keras.regularizers.l2(L2_weight_decay),
                             bias_regularizer=tf.keras.regularizers.l2(L2_weight_decay)),
        tf.keras.layers.GlobalMaxPooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=initializer),
    ]
    model = tf.keras.Sequential(layers)
    return model


learning_rate = 5e-3
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum = 0.9, nesterov=True)

model = model_init_fn()
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.binary_crossentropy,
              metrics=[tf.keras.metrics.binary_accuracy])


print(tf.shape(X_clean_train_correct))
print(tf.shape(X_clean_test_correct))
X_d_trian = tf.concat( [X_clean_train_correct, X_clean_test_correct], axis = 0)
print(tf.shape(X_d_trian))
#Y_d_trian = tf.concat(Y_clean_train_correct, Y_clean_test_correct)
Y_d_train = tf.ones(tf.shape(X_d_trian)[0])

# train detector and evaluate it
model.fit(X_d_trian, Y_d_train, batch_size=64, epochs=1) #, validation_data=(X_val, y_val))
model.evaluate(X_test, y_test)
