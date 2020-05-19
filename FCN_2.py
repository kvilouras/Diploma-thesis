import keras
import tensorflow

from keras.layers import Lambda, Input, SpatialDropout2D, concatenate, BatchNormalization, Conv2D, GlobalAveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.initializers import he_uniform, he_normal
from keras.regularizers import l2
from keras.utils import plot_model

x = Input(batch_shape=(None,128,431,1))

x1 = Lambda(lambda x : x[:,0:64,:,:])(x)
x2 = Lambda(lambda x : x[:,64:128,:,:])(x)

# low freqs
x1 = BatchNormalization()(x1)
x1 = Conv2D(data_format='channels_last', filters=16, kernel_size=(5, 5), activation='elu', padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.001))(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D(data_format='channels_last', padding='same', pool_size=(1, 10))(x1)
x1 = SpatialDropout2D(0.2)(x1)
x1 = Conv2D(data_format='channels_last', filters=32, kernel_size=(5, 5), activation='elu', padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.001))(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D(data_format='channels_last', padding='same', pool_size=(2, 5))(x1)
x1 = SpatialDropout2D(0.4)(x1)
x1 = Conv2D(data_format='channels_last', filters=64, kernel_size=(5, 5), activation='elu', padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.001))(x1)
x1 = BatchNormalization()(x1)
x1 = MaxPooling2D(data_format='channels_last', padding='same', pool_size=(1, 9))(x1)

# high freqs
x2 = BatchNormalization()(x2)
x2 = Conv2D(data_format='channels_last', filters=16, kernel_size=(5, 5), activation='elu', padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.001))(x2)
x2 = BatchNormalization()(x2)
x2 = MaxPooling2D(data_format='channels_last', padding='same', pool_size=(1, 10))(x2)
x2 = SpatialDropout2D(0.2)(x2)
x2 = Conv2D(data_format='channels_last', filters=32, kernel_size=(5, 5), activation='elu', padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.001))(x2)
x2 = BatchNormalization()(x2)
x2 = MaxPooling2D(data_format='channels_last', padding='same', pool_size=(2, 5))(x2)
x2 = SpatialDropout2D(0.4)(x2)
x2 = Conv2D(data_format='channels_last', filters=64, kernel_size=(5, 5), activation='elu', padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.001))(x2)
x2 = BatchNormalization()(x2)
x2 = MaxPooling2D(data_format='channels_last', padding='same', pool_size=(1, 9))(x2)

conc = concatenate([x1, x2])

conc = Conv2D(data_format='channels_last', filters=10, kernel_size=(1, 1), padding='same', kernel_initializer=he_normal(seed=42))(conc)
conc = BatchNormalization()(conc)
conc = GlobalAveragePooling2D(data_format='channels_last')(conc)
out = Activation('softmax')(conc)

model = Model(inputs=x, outputs=out)

model.summary()

plot_model(model, to_file='FCN_split_frequencies.png')

model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(learning_rate=0.0005),
              metrics = ['accuracy'])