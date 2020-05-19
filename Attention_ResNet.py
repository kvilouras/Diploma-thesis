### 1) Convolutional Block Attention Module (CBAM), doesn't perform well on this task ###
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, add, Lambda, Concatenate, multiply
from keras.initializers import he_normal
from keras import backend as K

def expand_dimensions(x):
    return K.expand_dims(x, -1)

def ChannelGAP(x):
    return K.mean(x, axis=-1)

def ChannelGMP(x):
    return K.max(x, axis=-1)

def CBAM(x, channels, reducted_units=2, block_name='CBAM_Block'):
    with K.name_scope(block_name):
        ### Channel Attention Module ###
        # input: conv layer feature maps (freq x time x channels)
        x_avg = GlobalAveragePooling2D(data_format='channels_last')(x)
        x_max = GlobalMaxPooling2D(data_format='channels_last')(x)

        shared_mlp1 = Dense(units=channels, kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))
        shared_mlp2 = Dense(units=reducted_units, kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001)) #r: reduction ratio=16 in paper and units=channels/ratio

        x_avg = shared_mlp2(x_avg)
        x_avg = Activation('relu')(x_avg)
        x_avg = shared_mlp1(x_avg)

        x_max = shared_mlp2(x_max)
        x_max = Activation('relu')(x_max)
        x_max = shared_mlp1(x_max)

        x_add = add([x_avg, x_max])
        x_add = Activation('sigmoid')(x_add)

        x2 = multiply([x_add, x]) # F' = Mc(F) * F

        ### Spatial Attention Module ###
        # apply pooling along the channel axis and concatenate -> generates feature descriptor
        chgap = Lambda(ChannelGAP)
        chgmp = Lambda(ChannelGMP)

        sp_avg = chgap(x2)
        sp_max = chgmp(x2)

        exp = Lambda(expand_dimensions)
        sp_avg = exp(sp_avg)
        sp_max = exp(sp_max)

        sp = Concatenate(axis=-1)([sp_max, sp_avg])

        sp = Conv2D(data_format='channels_last', filters=1, kernel_size=(7, 7), dilation_rate=(1,1), padding='same', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(sp)

        sp = Activation('sigmoid')(sp)

        x_out = multiply([sp, x2]) # F" = Ms(F') * F'

    return x_out

### 2) Squeeze and Excitation block (SE) ###
def SEblock(x, channels, reducted=1):
    x_avg = GlobalAveragePooling2D(data_format='channels_last')(x)
    dense_1 = Dense(units=channels, kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))
    dense_2 = Dense(units=reducted, kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))
    x_avg = dense_2(x_avg)
    x_avg = Activation('relu')(x_avg)
    x_avg = dense_1(x_avg)
    x_avg = Activation('sigmoid')(x_avg)
    x2 = multiply([x_avg, x])
    return x2

def residual_blockSE(X, f1, f2, red, filters, stage, block, drate, strides):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2 = filters
    X_shortcut = X
    X1 = Conv2D(filters=F1, kernel_size=(f1,f1), strides=strides, padding='same', activation='relu', dilation_rate=drate, name=conv_name_base + '2a', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X)
    X1 = BatchNormalization(name = bn_name_base + '2a')(X1)
    X1 = Conv2D(filters=F2, kernel_size=(f2,f2), strides=(1,1), padding='same', dilation_rate=drate, name=conv_name_base + '2b', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X1)

    X_shortcut = Conv2D(filters=F2, kernel_size=(1,1), strides=strides, dilation_rate=drate, padding='same', name=conv_name_base + '1', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X_shortcut)

    X1 = SEblock(X1, channels=F2, reducted=red)

    # final step
    X = Add()([X1, X_shortcut])
    X = Activation('relu')(X)
    X = BatchNormalization(name = bn_name_base + 'fin')(X)

    return X


def ResNet(input_shape=(128, 431, 1), classes=10):
    X_input = Input(input_shape)
    X = BatchNormalization(name='bn_1')(X_input)
    X = Conv2D(16, (5, 5), strides=(1, 1), name='conv1', padding='same', activation='relu',
               kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X)
    X = BatchNormalization(name='bn_conv1')(X)

    X = residual_blockSE(X, f1=5, f2=5, red=3, filters=[16, 24], stage=2, block='a', drate=(1, 1), strides=(1, 1))
    X = residual_blockSE(X, f1=5, f2=5, red=6, filters=[32, 48], stage=2, block='b', drate=(1, 1),
                         strides=(1, 1))  # r=8, works just fine

    X = residual_blockSE(X, f1=3, f2=3, red=6, filters=[64, 96], stage=3, block='a', drate=(1, 1), strides=(1, 1))
    X = residual_blockSE(X, f1=3, f2=3, red=12, filters=[128, 192], stage=3, block='b', drate=(1, 1),
                         strides=(1, 1))  # r=16, works just fine

    X = Conv2D(classes, (1, 1), strides=(1, 1), name='conv_last', padding='same', kernel_initializer=he_normal(seed=42),
               kernel_regularizer=l2(0.0001))(X)
    X = BatchNormalization(name='bn_conv_last')(X)
    X = GlobalAveragePooling2D()(X)
    X = Activation('softmax')(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet_SE')

    return model


model = ResNet(input_shape=(128, 431, 1), classes=10)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.0005),
              metrics=['accuracy'])

model.summary()