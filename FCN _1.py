import keras
import tensorflow

from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, SpatialDropout2D, GlobalAveragePooling2D, MaxPooling2D, Activation
from keras.optimizers import Adam, SGD
from keras.initializers import he_uniform, he_normal
from keras.regularizers import l2

# for PCEN input: kernel_regularizer=l2(0.0001)

model = Sequential([BatchNormalization(input_shape=(128, 431, 1)),
                    Conv2D(data_format='channels_last', filters=16, kernel_size=(5, 5), activation='elu',
                           padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.001)),
                    BatchNormalization(),
                    MaxPooling2D(data_format='channels_last', padding='same', pool_size=(1, 10)),
                    SpatialDropout2D(0.2),
                    Conv2D(data_format='channels_last', filters=32, kernel_size=(5, 5), activation='elu',
                           padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.001)),
                    BatchNormalization(),
                    MaxPooling2D(data_format='channels_last', padding='same', pool_size=(2, 5)),
                    SpatialDropout2D(0.3),
                    Conv2D(data_format='channels_last', filters=64, kernel_size=(5, 5), activation='elu',
                           padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.001)),
                    BatchNormalization(),
                    MaxPooling2D(data_format='channels_last', padding='same', pool_size=(2, 4)), 
                    SpatialDropout2D(0.4),
                    Conv2D(data_format='channels_last', filters=107, kernel_size=(3, 3), activation='elu',
                           padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.001)),
                    BatchNormalization(),
                    MaxPooling2D(data_format='channels_last', padding='same', pool_size=(1, 4)),

                    Conv2D(data_format='channels_last', filters=10, kernel_size=(1, 1), padding='same',
                           kernel_initializer=he_normal(seed=42)),
                    BatchNormalization(),
                    GlobalAveragePooling2D(data_format='channels_last'),
                    Activation('softmax'),
                    ])

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['accuracy'])

model.summary()

# Open the file
with open('report.txt', 'w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    model.summary(print_fn=lambda x: fh.write(x + '\n'))