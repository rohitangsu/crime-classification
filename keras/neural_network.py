import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import numpy as np
import pickle
import os

MODEL_SAVE_FILE = os.path.expanduser('~') + '/models/neural_network.model'


def get_data_from_file(filepath):
    if not os.path.exists(filepath):
        raise Exception('Data file path not found')
    data = pd.read_csv(filepath)
    return data

def split(data, proportion=0.7):
    msk = np.random.rand(len(data)) <= proportion
    train = data[msk]
    test = data[~msk]
    return train, test


def get_one_hot_encoded_target_columns(target):
    target = target.values
    encoder = LabelEncoder()
    encoder.fit(target)
    encoded_Y = encoder.transform(target)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y

# define baseline model
def baseline_model(input_dim= 60,num_output_variables=39):
    # create model
    model = Sequential()
    model.add(Dense(input_dim + num_output_variables, input_dim=input_dim, init='normal', activation='relu'))
    model.add(Dense(num_output_variables, init='normal', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def get_num_unique_targets(train_target):
    train_target_unique_value = train_target.unique()
    return train_target_unique_value.shape[0]

def get_num_feature(train):
    return train.shape[1]


def kcross_validation(estimator, train, target, seed):
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    results = cross_val_score(estimator, train, target, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

def train(self, filepath, target, feature_columns, seed):
    data = get_data_from_file(filepath)
    target_data = data[target]
    #getting the relevant features.
    data = data[feature_columns]
    train, test = split(data, 0.8)
    train_target, test_target = split(target_data, 0.8)

    #Set the seed for random function
    np.random.seed(seed)

    #Get one hot encoded data_set.
    one_hot_encoded_target = get_one_hot_encoded_target_columns(train_target)

    a, b = get_num_unique_targets(train_target), get_num_feature(train)

    estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)    
    
    #Fit the model 
    estimator.fit(train, target)
    pickle.dump(estimator, open(MODEL_SAVE_FILE, 'wb'))

    estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)
    #Do K fold cross Validation.
    kcross_validation(estimator, train, target, seed)







