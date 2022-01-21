import os
# Avoid memory explosion
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Omit Tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras import backend as K
# Input shape: [N x Length x 1]
K.set_image_data_format('channels_first')
from keras.models import load_model
from keras.optimizers import Adam, Adamax, RMSprop
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from models.awf import AWFNet
from models.df import DFNet
from models.var_cnn import VarCNN

from data_utils import load_sirinam_dataset
from data_utils import load_rimmer_dataset


def train_model(target_model, dataset):
    """
    Generate perturbation for traces of a website.
    """
    model_dir = './checkpoint/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    saved_model = model_dir + '%s_%s' % (target_model.lower(), dataset.lower())

    if target_model.lower() == 'awf':
        input_size = 3000
    elif target_model.lower() == 'df':
        input_size = 5000
    elif target_model.lower() == 'varcnn':
        input_size = 5000
    
    print('==> Loading %s dataset...' % dataset)
    if dataset.lower().startswith('sirinam'):
        num_classes = 95
        X_train, y_train, X_valid, y_valid, X_test, y_test = \
            load_sirinam_dataset(input_size, num_classes)

    elif dataset.lower().startswith('rimmer'):
        num_classes = int(dataset[6:])
        X_train, y_train, X_valid, y_valid, X_test, y_test = \
            load_rimmer_dataset(input_size, num_classes)

    print('Training data shape:', X_train.shape)
    print('Validation data shape:', X_valid.shape)
    print('Testing data shape:', X_test.shape)
    
    if not os.path.exists('%s.h5' % saved_model):
        
        print('==> Building %s model...' % target_model)
        if target_model.lower() == 'awf':
            model = AWFNet.build((input_size, 1), num_classes)
            optimizer = RMSprop
            learning_rate = 0.0011
            batch_size = 256
            max_epochs = 30
            patience = 5

        elif target_model.lower() == 'df':
            model = DFNet.build((input_size, 1), num_classes)
            optimizer = Adamax
            learning_rate = 0.002
            batch_size = 128
            max_epochs = 50
            patience = 5

        elif target_model.lower() == 'varcnn':
            model = VarCNN.build((input_size, 1), num_classes)
            optimizer = Adam
            learning_rate = 0.001
            batch_size = 128
            max_epochs = 80
            patience = 10
        
        model.compile(loss='categorical_crossentropy',
            optimizer=optimizer(learning_rate),
            metrics=['acc'])

        callbacks = [CSVLogger('%s.csv' % saved_model),
                    EarlyStopping(monitor='val_loss', patience=patience), 
                    ModelCheckpoint('%s.h5' % saved_model, monitor='val_loss',
                        save_best_only=True)]

        if target_model.lower() == 'varcnn':
            callbacks.append(ReduceLROnPlateau(monitor='val_loss',
                factor=0.316, patience=patience//2, cooldown=0, min_lr=1e-5))

        model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=max_epochs,
            callbacks=callbacks,
            validation_data=(X_valid, y_valid))

    print('==> Loading %s model...' % target_model)
    model = load_model('%s.h5' % saved_model)
    
    if not os.path.exists('%s.sum' % saved_model):
        with open('%s.sum' % saved_model, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    model.compile(loss='categorical_crossentropy', metrics=['acc'])
    pred_class = model.predict(X_test).argmax(axis=-1)
    true_class = y_test.argmax(axis=-1)
    result = model.evaluate(X_test, y_test)

    print('Test loss: %.4f' % result[0])
    print('Accuracy: %.2f%% (%d/%d)' % (float(result[1] * 100),
        sum(pred_class == true_class), len(y_test)))
    