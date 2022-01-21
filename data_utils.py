import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical


def train_test_valid_split(X, y, valid_size=0.1, test_size=0.1):
    """
    Split data into training, validation, and test sets.
    Set random_state=0 to keep the same split.
    """
    # Split into training set and others
    split_size = valid_size + test_size
    [X_train, X_, y_train, y_] = train_test_split(X, y,
                                    test_size=split_size,
                                    random_state=0,
                                    stratify=y)

    # Split into validation set and test set
    split_size = test_size / (valid_size + test_size)
    [X_valid, X_test, y_valid, y_test] = train_test_split(X_, y_,
                                            test_size=split_size,
                                            random_state=0,
                                            stratify=y_)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def format_data(X, y, input_size, num_classes):
    """
    Format traces into input shape [N x Length x 1] and one-hot encode labels.
    """
    X = X[:, :input_size]
    X = X.astype('float32')
    X = X[:, :, np.newaxis]

    y = y.astype('int32')
    y = to_categorical(y, num_classes)

    return X, y


def format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, input_size, num_classes):
    X_train, y_train = format_data(X_train, y_train, input_size, num_classes)
    X_valid, y_valid = format_data(X_valid, y_valid, input_size, num_classes)
    X_test, y_test = format_data(X_test, y_test, input_size, num_classes)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_sirinam_dataset(input_size=5000, num_classes=95, formatting=True):
    """
    Load Sirinam's (CCS'18) dataset.
    """
    # Point to the directory storing data
    dataset_dir = './data/Sirinam/'

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle, encoding='latin1'))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle, encoding='latin1'))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle, encoding='latin1'))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle, encoding='latin1'))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle, encoding='latin1'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle, encoding='latin1'))

    if formatting:
        return format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, input_size, num_classes)
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_rimmer_dataset(input_size=5000, num_classes=100, formatting=True):
    """
    Load Rimmer's (NDSS'18) dataset.
    """
    # Point to the directory storing data
    dataset_dir = './data/Rimmer/'

    # Load data
    datafile = dataset_dir + 'tor_%dw_2500tr.npz' % num_classes
    with np.load(datafile, allow_pickle=True) as npzdata:
        data = npzdata['data']
        labels = npzdata['labels']

    # Convert website to integer
    y = labels.copy()
    websites = np.unique(labels)
    for w in websites:
        y[np.where(labels == w)] = np.where(websites == w)[0][0]

    # Split data to fixed parts
    X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid_split(data, y, 0.05, 0.05)
    with open(dataset_dir + 'tor_%dw_2500tr_test.npz' % num_classes, 'wb') as handle:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, handle)

    if formatting:
        return format_data_all(X_train, y_train, X_valid, y_valid, X_test, y_test, input_size, num_classes)
    else:
        return X_train, y_train, X_valid, y_valid, X_test, y_test


def load_rimmer_websites(num_classes=100):
    """
    Load website labels of Rimmer's dataset.
    """
    # Point to the directory storing data
    dataset_dir = './data/Rimmer/'

    # Load data
    datafile = dataset_dir + 'tor_%dw_2500tr.npz' % num_classes
    with np.load(datafile, allow_pickle=True) as npzdata:
        labels = npzdata['labels']

    labels = np.unique(labels)

    # Save labels
    labelfile = dataset_dir + 'tor_%dw_labels.csv' % num_classes
    pd.Series(labels).to_csv(labelfile)

    return labels


def load_rimmer_concept_drift_data(time_gap, names, input_size=5000, formatting=True):
    """
    Load Rimmer's conecpt drift dataset.
    """
    # Point to the directory storing data
    dataset_dir = './data/Rimmer/'

    # Load data
    datafile = dataset_dir + 'tor_time_test%s_200w_100tr.npz' % time_gap
    with np.load(datafile, allow_pickle=True) as npzdata:
        traces = npzdata['data']
        labels = npzdata['labels']
    X_test = []
    y_test = []
    for trace, label in zip(traces, labels):
        # Ignore sites not in the 200w dataset
        if label in names:
            X_test.append(trace)
            y_test.append(np.where(names==label)[0][0])
    X_test = np.stack(X_test)
    y_test = np.array(y_test)

    if formatting:
        X_test, y_test = format_data(X_test, y_test, input_size, 200)
    
    return X_test, y_test
