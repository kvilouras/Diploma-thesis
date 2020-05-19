import tensorflow as tf
import keras

from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, GlobalAveragePooling2D, Dense, MaxPooling2D, Input, Add, Activation
from keras.layers import Lambda, concatenate
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.initializers import he_uniform, he_normal
from keras.regularizers import l2

from keras import backend as K
from keras.layers import Layer

class ShakeShake(Layer):

    def __init__(self, **kwargs):
        super(ShakeShake, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ShakeShake, self).build(input_shape)

    def call(self, x):
        # unpack x1 and x2
        assert isinstance(x, list)
        x1, x2 = x

        # create alpha and beta
        batch_sizes = K.shape(x1)[0]
        alpha = tf.keras.backend.random_uniform(shape=(batch_sizes, 1, 1, 1)) # error occurs with K.random_uniform
        beta = tf.keras.backend.random_uniform(shape=(batch_sizes, 1, 1, 1))

        # shake-shake during training phase
        def x_shake():
            return beta * x1 + (1 - beta) * x2 + K.stop_gradient((alpha - beta) * x1 + (beta - alpha) * x2)
        # even-even during testing phase
        def x_even():
            return 0.5 * x1 + 0.5 * x2
        return K.in_train_phase(x_shake, x_even)

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]


def residual_block(X, f1, f2, filters, stage, block, drate, strides):
    # define name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2 = filters

    # store input
    X_shortcut = X

    # first component of main path
    X1 = Conv2D(filters=F1, kernel_size=(f1,f1), strides=strides, padding='same', activation='relu', dilation_rate=drate, name=conv_name_base + '2a', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X)
    X1 = BatchNormalization(name = bn_name_base + '2a')(X1)

    # second component of main path
    X1 = Conv2D(filters=F2, kernel_size=(f2,f2), strides=(1,1), padding='same', dilation_rate=drate, name=conv_name_base + '2b', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X1)

    # first component of secondary path
    X2 = Conv2D(filters=F1, kernel_size=(f1,f1), strides=strides, padding='same', activation='relu', dilation_rate=drate, name=conv_name_base + '2c', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X)
    X2 = BatchNormalization(name = bn_name_base + '2c')(X2)

    # second component of secondary path
    X2 = Conv2D(filters=F2, kernel_size=(f2,f2), strides=(1,1), padding='same', dilation_rate=drate, name=conv_name_base + '2d', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X2)

    # Skip connection: transform input in order to have same dimensions as X
    X_shortcut = Conv2D(filters=F2, kernel_size=(1,1), strides=strides, dilation_rate=drate, padding='same', name=conv_name_base + '1', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X_shortcut)

    # final step
    X = Add()([X_shortcut, ShakeShake()([X1, X2])])
    X = Activation('relu')(X)
    X = BatchNormalization(name = bn_name_base + 'fin')(X)

    return X

def ResNet(input_shape=(128,431,1), classes=10):

    X_input = Input(input_shape)

    # stage 1
    X = BatchNormalization(name='bn_1')(X_input)
    X = Conv2D(16, (5,5), strides=(1,1), name='conv1', padding='same', activation='relu', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X)
    X = BatchNormalization(name='bn_conv1')(X)
    X = MaxPooling2D(padding='same', pool_size=(1,10))(X)

    # stage 2
    X = residual_block(X, f1=5, f2=5, filters=[24, 32], stage=2, block='a', drate=(1,1), strides=(1,1))
    X = MaxPooling2D(padding='same', pool_size=(2,5))(X)

    # stage 3
    X = residual_block(X, f1=5, f2=5, filters=[48, 64], stage=3, block='a', drate=(1,1), strides=(1,1))
    X = MaxPooling2D(padding='same', pool_size=(2,3))(X)

    # stage 4
    X = residual_block(X, f1=3, f2=3, filters=[96, 128], stage=4, block='a', drate=(1,1), strides=(1,1))
    X = MaxPooling2D(padding='same', pool_size=(1,3))(X)

    # stage 5
    X = residual_block(X, f1=3, f2=1, filters=[192, 128], stage=5, block='a', drate=(1,1), strides=(1,1))

    # output
    X = Conv2D(classes, (1,1), name='conv_last', padding='same', kernel_initializer=he_normal(seed=42))(X)
    X = BatchNormalization(name='bn_conv_last')(X)
    X = GlobalAveragePooling2D()(X)
    X = Activation('softmax')(X)

    # create model
    model = Model(inputs = X_input, outputs = X, name='ResNet')

    return model

model = ResNet(input_shape=(128,431,1), classes=10)

model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(learning_rate=0.0005),
              metrics = ['accuracy'])

with open('ResNet.txt','w') as fh:
    #Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))