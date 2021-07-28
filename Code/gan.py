######################################################################## Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
import tensorflow as tf
# %tensorflow_version 1.x
import warnings
import joblib
warnings.filterwarnings("ignore")
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam
#from keras.utils import to_categorical                ##########################################################################
from keras.layers.advanced_activations import LeakyReLU
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
import scipy.stats
################################################################# Read CSV files
result = pd.read_csv("../Data/Merged_Shuffled_Datasets.csv")
########################################################### MinMax Normalization
mms = MinMaxScaler()
numerical_data_rescaled = mms.fit_transform(result)
numerical_data_rescaled
################################################################# Generator Func 
def build_generator(n_columns, latent_dim):
    model = Sequential()
    model.add(Dense(32, kernel_initializer = "he_uniform", input_dim=latent_dim))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(64,  kernel_initializer = "he_uniform"))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(128,  kernel_initializer = "he_uniform"))
    model.add(LeakyReLU(0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(n_columns, activation = "sigmoid"))
    return model
############################################################# Discriminator Func
def build_discriminator(inputs_n):
	model = Sequential()
	model.add(Dense(128,  kernel_initializer = "he_uniform", input_dim = inputs_n))
	model.add(LeakyReLU(0.2))
	model.add(Dense(64,  kernel_initializer = "he_uniform"))
	model.add(LeakyReLU(0.2))
	model.add(Dense(32,  kernel_initializer = "he_uniform"))
	model.add(LeakyReLU(0.2))
	model.add(Dense(16,  kernel_initializer = "he_uniform"))
	model.add(LeakyReLU(0.2))
	model.add(Dense(1, activation = "sigmoid"))
	model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ["accuracy"])
	return model
################################################################ Build generator
latent_dim = 4
generator = build_generator(numerical_data_rescaled.shape[1], latent_dim)
plot_model(generator, show_layer_names = True, show_shapes = True)
################################################################## Set optimizer
optimizer = Adam(lr=0.0002, beta_1=0.5)
############################################################ Build discriminator
discriminator = build_discriminator(numerical_data_rescaled.shape[1])
plot_model(discriminator, show_layer_names = True, show_shapes = True)
####################################################################### GAN Func
def build_gan(generator, discriminator):
	discriminator.trainable = False
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	model.compile(loss = "binary_crossentropy", optimizer = optimizer)
	return model
###################################################################### Build gan
gan = build_gan(generator, discriminator)
plot_model(gan, show_layer_names = True, show_shapes = True)
#################################################################### Train model
def train(gan, generator, discriminator, data, latent_dim, n_epochs, n_batch, n_eval):
    #i used Half batch size for updateting discriminator
    half_batch = int(n_batch / 2)
    generator_loss = []
    discriminator_loss = []
    #generate class labels for fake = 0 and real = 1
    valid = np.ones((half_batch, 1))
    fake = np.zeros((half_batch, 1))
    y_gan = np.ones((n_batch, 1))
    for i in range(n_epochs):
        #select random batch from the real data
        idx = np.random.randint(0, data.shape[0], half_batch)
        real_data = data[idx]
        #generate fake samples from the noise
        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        fake_data = generator.predict(noise)
        #train 
        d_loss_real, _ = discriminator.train_on_batch(real_data, valid)
        d_loss_fake, _ = discriminator.train_on_batch(fake_data, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        discriminator_loss.append(d_loss)
        #generate noise for generator input and  we train the generator ( i used this to have the discriminator label samples as valid)
        noise = np.random.normal(0, 1, (n_batch, latent_dim))
        g_loss = gan.train_on_batch(noise, y_gan)
        generator_loss.append(g_loss)
        #evaluate
        if (i+1) % n_eval == 0:
            print ("Epoch: %d [Generator loss: %f] [Discriminator loss: %f]" % (i + 1, g_loss, d_loss))
    plt.figure(figsize = (20, 10))
    plt.plot(generator_loss, label = "Generator loss")
    plt.plot(discriminator_loss, label = "Discriminator loss")
    plt.title("Stats from training GAN")
    plt.legend()
    plt.grid()
#Train and plot loss values
train(gan, generator, discriminator, numerical_data_rescaled, latent_dim, n_epochs = 1000, n_batch = 1024, n_eval = 250)
#Save model
generator.save('../Model/genrator_model.h5')
#Save minmax features
joblib.dump(mms, '../Model/scaler.gz')
my_scaler = joblib.load('../Model/scaler.gz')