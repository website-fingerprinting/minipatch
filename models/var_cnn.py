from keras.models import Model
from keras.layers import Input, ZeroPadding1D, Conv1D, BatchNormalization, \
    Activation, MaxPooling1D, Add, GlobalAveragePooling1D, Dense


def dilated_basic_block_1d(filters, layer, block, dilations):
    """
    A one-dimensional basic residual block with dilations.

    :param filters: the output's feature space
    :param layer: int representing the layer of this block (starting from 2)
    :param block: int representing this block (starting from 1)
    :param dilations: tuple representing amount to dilate first and second conv
    """
    label = 'layer{}_block{}'.format(layer, block)

    if layer == 2 or block != 1:
        stride = 1
    else:
        stride = 2

    def f(x):
        y = Conv1D(filters=filters, kernel_size=3, strides=stride, padding='causal', use_bias=False,
                dilation_rate=dilations[0], kernel_initializer='he_normal', name='%s_conv1' % label)(x)
        y = BatchNormalization(epsilon=1e-5, name='%s_bn1' % label)(y)
        y = Activation('relu', name='%s_relu1' % label)(y)

        y = Conv1D(filters=filters, kernel_size=3, padding='causal', use_bias=False,
                dilation_rate=dilations[1], kernel_initializer='he_normal', name='%s_conv2' % label)(y)
        y = BatchNormalization(epsilon=1e-5, name='%s_bn2' % label)(y)

        if layer > 2 and block == 1:
            shortcut = Conv1D(filters=filters, kernel_size=1, strides=stride, use_bias=False,
                            kernel_initializer='he_normal', name='%s_ds_conv' % label)(x)
            shortcut = BatchNormalization(epsilon=1e-5, name='%s_ds_bn' % label)(shortcut)
        else:
            shortcut = x

        y = Add(name='%s_add' % label)([y, shortcut])
        y = Activation('relu', name='%s_relu2' % label)(y)
        return y

    return f


class VarCNN:
    @staticmethod
    def build(input_shape, classes):
        """
        ResNet18
        """
        # The network's residual architecture
        layer_blocks = [2, 2, 2, 2]

        input = Input(shape=input_shape, name='input')
        x = ZeroPadding1D(padding=3, name='layer1_padding')(input)
        x = Conv1D(filters=64, kernel_size=7, strides=2, use_bias=False, name='layer1_conv')(x)
        x = BatchNormalization(epsilon=1e-5, name='layer1_bn')(x)
        x = Activation('relu', name='layer1_relu')(x)
        x = MaxPooling1D(pool_size=3, strides=2, padding='same', name='layer1_pool')(x)

        features = 64
        outputs = []
        for i, blocks in enumerate(layer_blocks):
            x = dilated_basic_block_1d(features, i+2, 1, dilations=(1, 2))(x)
            for block in range(2, blocks+1):
                x = dilated_basic_block_1d(features, i+2, block, dilations=(4, 8))(x)
            features *= 2
            outputs.append(x)
        x = GlobalAveragePooling1D(name='average_pool')(x)

        output = Dense(units=classes, activation='softmax', name='output')(x)

        model = Model(inputs=input, outputs=output, name='VarCNN')

        return model
