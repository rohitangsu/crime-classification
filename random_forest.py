
""" ------------------------------ ABOUT THE DATA --------------------------------------------
    The data consists of the following colums..

    Category: WARRANTS,   OTHER OFFENSES, LARCENY/THEFT, VEHICLE THEFT , ................ 39 terms
    Independent Variables:
      1. 'Dates' 
      2. 'Category'
      3. 'Descript'
      4. 'DayOfWeek'
      5. 'PdDistrict' 
      6. 'Resolution'
      7. 'Address'
      8.  'X', '
      9.   Y'

    The Categorical variables are already one-hot encoded for ease
    The idea is to train a Random Forest Classifier to classify a set of features 
    to a crime category 
-------------------------------------------------------------------------------------------"""

from sklearn.ensemble import  RandomForestClassifier
from collections import defaultdict
from numpy import array
import pandas as pd
import numpy as np
import pickle
import os

MODEL_SAVE_FILE = os.path.expanduser('~') + '/models/randomforestclassifier.model'

def split(dataset, proportion=0.7):
  msk = np.random.rand(len(dataset)) < proportion
  train = dataset[msk]
  test = dataset[~msk]
  return train, test

def enumerate_class_names(classes):
  number_of_unique_classes = classes.shape[0]
  index = 0
  class_name_to_index_dict = defaultdict(int)
  for classname in number_of_unique_classes:
    class_name_to_index_dict[classname] = index
    index += 1
  return class_name_to_index_dict


def get_indexed_numpy_array(class_data):
  unique_classes = class_data.unique
  class_name_to_index_dict = enumerate_class_names(unique_classes)
  class_array = list()
  for classname in class_data:
     index = class_name_to_index_dict[classname]
     class_array.append(index)
  return array(class_array)

def train(filepath, category_name='Category'):
  #Read csv
  dataset = pd.read_csv(filepath)
  
  #Relevant feature columns start from 8:
  dataset = dataset.ix[:, 8:]

  #target columns
  target = dataset[category_name]
  #split into test and train
  train, test = split(dataset, 0.8)
  train_label, test_label = split(target, 0.8) 
  #Get indexed category numpy array
  train_label = get_indexed_numpy_array(train)
  #Converting dataframe to numpy array object
  train = train.values
  #train the model
  clf = RandomForestClassifier(n_jobs=2) 
  clf.fit(train, train_label)
  #Save model to file
  pickle.dump(clf, open(MODEL_SAVE_FILE, 'wb'))
  #Do feature importance calculation
  feature_importance(clf.feature_importances_, target.shape[1]) 
  # Evaluate
  print(evaluate(clf, test, test_label))


def evaluate(model, test, test_label):
  pass

def feature_importance(importances, size):
  indices = np.argsort(importances)[::-1]
  for f in range(size):
      print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
 
  