import os
# Avoid memory explosion
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# Omit Tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle

from keras import backend as K
# Input shape: [N x Length x 1]
K.set_image_data_format('channels_first')
from keras.models import load_model

from data_utils import load_sirinam_dataset
from data_utils import load_rimmer_dataset, load_rimmer_websites
from data_utils import load_rimmer_concept_drift_data


def load_trained_model(model, dataset, compile=False):
    saved_model = './checkpoint/%s_%s' % (model.lower(), dataset.lower())
    if not os.path.exists('%s.h5' % saved_model):
        raise Exception('Please train %s on %s first' % (model, dataset))
    
    model = load_model('%s.h5' % saved_model, compile=False)
    
    if compile:
        model.compile(loss='categorical_crossentropy', metrics=['acc'])

    return model


def load_data(dataset, input_size, num_classes, data, verbose=1):
    if dataset.lower().startswith('sirinam'):
        names = None
        if data == 'test':
            _, _, _, _, traces, labels = load_sirinam_dataset(input_size, num_classes)
        elif data == 'valid':
            _, _, traces, labels, _, _ = load_sirinam_dataset(input_size, num_classes)
        else:
            raise Exception('Sirinam supports only test and valid data')

    elif dataset.lower().startswith('rimmer'):
        names = load_rimmer_websites(num_classes)
        if data == 'test':
            _, _, _, _, traces, labels = load_rimmer_dataset(input_size, num_classes)
        elif data == 'valid':
            _, _, traces, labels, _, _ = load_rimmer_dataset(input_size, num_classes)
        else:
            if num_classes != 200:
                raise Exception('Conecpt drift dataset should be Rimmer200')
            if data not in ['3d', '10d', '2w', '4w', '6w']:
                raise Exception('Conecpt drift dataset supports only 3d/10d/2w/4w/6w')
            traces, labels = load_rimmer_concept_drift_data(data, names, input_size)

    if verbose > 0:
        print('Data shape:', traces.shape)
    
    return traces, labels, names


def save_checkpoint(perturber, checkpoint, results, filename):
    """
    Save partial results.
    """
    data = {'model': perturber.model.name,
        'input': perturber.input_size,
        'classes': perturber.num_classes,
        'checkpoint': checkpoint,
        'results': results}

    with open('%s.pkl' % filename, 'wb') as handle:
        pickle.dump(data, handle)


def load_checkpoint(perturber, filename):
    """
    Load and check partial results.
    """
    if not os.path.exists('%s.pkl' % filename):
        return -1, []

    with open('%s.pkl' % filename, 'rb') as handle:
        data = pickle.load(handle)

    assert data['model'] == perturber.model.name
    assert data['input'] == perturber.input_size
    assert data['classes'] == perturber.num_classes
    return data['checkpoint'], data['results']


def del_checkpoint(filename):
    """
    Delete partial results.
    """
    os.remove('%s.pkl' % filename)
