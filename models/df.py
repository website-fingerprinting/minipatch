from keras.models import Sequential
from keras.layers import Conv1D, BatchNormalization, ELU, \
    MaxPooling1D, Dropout, Activation, Flatten, Dense
from keras.initializers import glorot_uniform

class DFNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential(name='DF')

        filter_num = ['None',32,64,128,256]
        kernel_size = ['None',8,8,8,8]
        conv_stride_size = ['None',1,1,1,1]
        pool_stride_size = ['None',4,4,4,4]
        pool_size = ['None',8,8,8,8]
        
        model.add(Conv1D(input_shape=input_shape,
                         filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv1'))
        model.add(BatchNormalization(name='block1_bn1'))
        model.add(ELU(alpha=1.0, name='block1_elu1'))
        model.add(Conv1D(filters=filter_num[1], kernel_size=kernel_size[1],
                         strides=conv_stride_size[1], padding='same',
                         name='block1_conv2'))
        model.add(BatchNormalization(name='block1_bn2'))
        model.add(ELU(alpha=1.0, name='block1_elu2'))
        model.add(MaxPooling1D(pool_size=pool_size[1], strides=pool_stride_size[1],
                               padding='same', name='block1_pool'))
        model.add(Dropout(rate=0.1, name='block1_dropout'))

        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv1'))
        model.add(BatchNormalization(name='block2_bn1'))
        model.add(Activation('relu', name='block2_relu1'))
        model.add(Conv1D(filters=filter_num[2], kernel_size=kernel_size[2],
                         strides=conv_stride_size[2], padding='same',
                         name='block2_conv2'))
        model.add(BatchNormalization(name='block2_bn2'))
        model.add(Activation('relu', name='block2_relu2'))
        model.add(MaxPooling1D(pool_size=pool_size[2], strides=pool_stride_size[3],
                               padding='same', name='block2_pool'))
        model.add(Dropout(rate=0.1, name='block2_dropout'))

        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv1'))
        model.add(BatchNormalization(name='block3_bn1'))
        model.add(Activation('relu', name='block3_relu1'))
        model.add(Conv1D(filters=filter_num[3], kernel_size=kernel_size[3],
                         strides=conv_stride_size[3], padding='same',
                         name='block3_conv2'))
        model.add(BatchNormalization(name='block3_bn2'))
        model.add(Activation('relu', name='block3_relu2'))
        model.add(MaxPooling1D(pool_size=pool_size[3], strides=pool_stride_size[3],
                               padding='same', name='block3_pool'))
        model.add(Dropout(rate=0.1, name='block3_dropout'))

        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv1'))
        model.add(BatchNormalization(name='block4_bn1'))
        model.add(Activation('relu', name='block4_relu1'))
        model.add(Conv1D(filters=filter_num[4], kernel_size=kernel_size[4],
                         strides=conv_stride_size[4], padding='same',
                         name='block4_conv2'))
        model.add(BatchNormalization(name='block4_bn2'))
        model.add(Activation('relu', name='block4_relu2'))
        model.add(MaxPooling1D(pool_size=pool_size[4], strides=pool_stride_size[4],
                               padding='same', name='block4_pool'))
        model.add(Dropout(rate=0.1, name='block4_dropout'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(units=512, kernel_initializer=glorot_uniform(seed=0), name='fc1'))
        model.add(BatchNormalization(name='fc1_bn'))
        model.add(Activation('relu', name='fc1_relu'))

        model.add(Dropout(rate=0.7, name='fc1_dropout'))

        model.add(Dense(units=512, kernel_initializer=glorot_uniform(seed=0), name='fc2'))
        model.add(BatchNormalization(name='fc2_bn'))
        model.add(Activation('relu', name='fc2_relu'))

        model.add(Dropout(rate=0.5, name='fc2_dropout'))

        model.add(Dense(units=classes, kernel_initializer=glorot_uniform(seed=0),
                        activation='softmax', name='fc3'))

        return model
