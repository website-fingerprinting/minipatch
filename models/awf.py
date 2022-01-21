from keras.models import Sequential
from keras.layers import Dropout, Conv1D, MaxPooling1D, Flatten, Dense

class AWFNet:
    @staticmethod
    def build(input_shape, classes):
        model = Sequential(name='AWF')

        dropout = 0.1
        filters = 32
        kernel_size = 5
        stride_size = 1
        pool_size = 4

        model.add(Dropout(input_shape=input_shape,
                        rate=dropout, name='dropout'))

        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                        strides=stride_size, padding='valid',
                        activation='relu', name='block1_conv'))
        model.add(MaxPooling1D(pool_size=pool_size, padding='valid',
                        name='block1_pool'))

        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                        strides=stride_size, padding='valid',
                        activation='relu', name='block2_conv'))
        model.add(MaxPooling1D(pool_size=pool_size, padding='valid',
                        name='block2_pool'))

        model.add(Conv1D(filters=filters, kernel_size=kernel_size,
                        strides=stride_size, padding='valid',
                        activation='relu', name='block3_conv'))
        model.add(MaxPooling1D(pool_size=pool_size, padding='valid',
                        name='block3_pool'))

        model.add(Flatten(name='flatten'))
        model.add(Dense(units=classes, activation='softmax', name='dense'))

        return model
