### replace ResNet from the original script 'ResNet.py' with the following: ###

# ResNet with split frequency paths

def ResNet(input_shape=(128,431,1), classes=10):

    X_input = Input(input_shape)
    X_low = Lambda(lambda x: x[:,0:64,:,:])(X_input)
    X_high = Lambda(lambda x: x[:,64:128,:,:])(X_input)

    X_low = BatchNormalization(name='bn_1_low')(X_low)
    X_high = BatchNormalization(name='bn_1_high')(X_high)

    # stage 1, low freqs
    X_low = Conv2D(16, (5,5), strides=(1,1), name='conv1_low', padding='same', activation='relu', kernel_initializer=he_normal(seed=42), kernel_regularizer=l2(0.0001))(X_low)
    X_low = BatchNormalization(name='bn_conv1_low')(X_low)
    X_low = MaxPooling2D(padding='same', pool_size=(1,10))(X_low)

    # stage 2, low freqs
    X_low = residual_block(X_low, f1=5, f2=5, filters=[24, 32], stage=2, block='a_low', drate=(1,1), strides=(1,1))
    X_low = MaxPooling2D(padding='same', pool_size=(2,5))(X_low)

    # stage 3, low freqs
    X_low = residual_block(X_low, f1=5, f2=5, filters=[48, 64], stage=3, block='a_low', drate=(1,1), strides=(1,1))
    X_low = MaxPooling2D(padding='same', pool_size=(2,9))(X_low)

    # stage 1, high freqs
    X_high = Conv2D(16, (5,5), strides=(1,1), name='conv1_high', padding='same', activation='relu', kernel_initializer=he_uniform(seed=42), kernel_regularizer=l2(0.0001))(X_high)
    X_high = BatchNormalization(name='bn_conv1_high')(X_high)
    X_high = MaxPooling2D(padding='same', pool_size=(1,10))(X_high)

    # stage 2, high freqs
    X_high = residual_block(X_high, f1=5, f2=5, filters=[24, 32], stage=4, block='a_high', drate=(1,1), strides=(1,1))
    X_high = MaxPooling2D(padding='same', pool_size=(2,5))(X_high)

    # stage 3, high freqs
    X_high = residual_block(X_high, f1=5, f2=5, filters=[48, 64], stage=5, block='a_high', drate=(1,1), strides=(1,1))
    X_high = MaxPooling2D(padding='same', pool_size=(2,9))(X_high)

    # merge low and high freqs
    X = concatenate([X_low, X_high])

    X = residual_block(X, f1=3, f2=3, filters=[96, 128], stage=6, block='a', drate=(1,1), strides=(1,1))

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

model.summary()
plot_model(model, to_file='ResNet_split_frequencies.png')