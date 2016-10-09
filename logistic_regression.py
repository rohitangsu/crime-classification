import pandas as pd
import numpy as np
import statsmodel.api as sm
from numpy import array
from matplotlib import pylab
import pylab as pl
import os

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
    The idea is to train a multiclass Logistic Regression Classifier.
-------------------------------------------------------------------------------------------"""



#Read the dataset into the memory as a panda dataframe.
DATA_DIRECTORY = os.path.expanduser('~') + '/data/'

if os.path.exists(DATA_DIRECTORY):
    #Raise exception.
    raise Exception('DATA DIRECTORY NOT FOUND')
#CRIME CLASSIFICATION DIRECTORY
FILE_DIRECTORY = os.path.join(DATA_DIRECTORY, 'crime.csv')

if os.path.exists(FILE_DIRECTORY):
    #Raise exception
    raise Exception('FILE NOT FOUND')

crime_data = pd.read_csv(FILE_DIRECTORY)

TARGET_COLUMN = 'Category'

category = crime_data[TARGET_COLUMN]

#Divide it into train set and test set.
msk = msk = np.random.rand(len(crime_data)) < 0.8
train = crime_data[msk]
train_labels = category[msk]

unique_category_names = train_labels.unique()
train_labels_to_np = train_labels.values
category_to_label_dict = dict()
#Convert the category data into category index values so that the fit function can fit.
for i, category in enumerate(unique_category_names):
    category_to_label_dict[category] = i
new_train_labels_int = list()
for category in train_labels_to_np:
    new_train_labels_int.append(category_to_label_dict[category])

new_train_labels_int = array(new_train_labels_int)
new_train_labels_int = new_train_labels_int / unique_category_names.shape[0]

# Train the logit function
try:
    logit = sm.Logit(new_train_labels_int, train.values)
    logit.fit()
    print('Train data Fitted.')
except Exception as e:
    print('Error while fitting the logit function.')
    print('Exact Error: ', + str(e))





"""  There are in total unique_category_name.shape[0] classes 

    n =  unique_category_name.shape[0]
    Hence, we shall divide the entire [0,1] interval into n classes.
    so, for those points falling in the interval i/n, i+1/n , class label would be i  
"""

#Evaluate the model.


#Get the test set and calculate the number of correct classification, and the accuracy measure.