import yaml
import argparse
from preprocess import *
from model import *
from data_partitioning import *


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
import numpy as np
import math


parser = argparse.ArgumentParser(description='Configure Training code')
parser.add_argument('--with-yaml', dest='yaml', action='store_true')

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

def unbalance_dataset(x_train, y_train):

    sm = SMOTE()
    x_res, y_res = sm.fit_resample(x_train, y_train)

    return x_res, y_res



if __name__ == "__main__":

    args = parser.parse_args()
    
    config = None
    if(args.yaml):
        f_name = 'config.yaml'

        with open('config.yaml') as file:
            config = yaml.safe_load(file)
    else:
        #Initialise defaults if yaml not used
        config = {
            'DATASET': 'chess.data',
            'LEARNING_RATE': 0.1,
            'DATASPLIT': 0.33,
            'PROBLEM': "Classification",
            'DATA_SPLIT': 0.1
        }
    
    stub = True

    data = read_data(config['DATASET'])
    
    print("ML Problem Type: {}".format(config['PROBLEM']))
    
    # Convert strings to numbers
    encoded_letters = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8
    }



    
    categorical_features = ['king_x', 'rook_x', 'king2_x']
    for feature in categorical_features:
        for letter in encoded_letters:
            data[feature] = data[feature].replace(letter, encoded_letters[letter])
    # X contains everything except label
    features = data.iloc[:, :-1]
    features = normalize_tabular_data(features)
    targets = label_to_number(data, 'sucess')
    
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=config['DATA_SPLIT'], random_state=42)

    if config['SMOTE'] == True:
        
        x_train, y_train = unbalance_dataset(x_train, y_train)

    model1 = model(x_train, x_test, y_train, y_test, DecisionTreeClassifier(), "Decision_Tree")
    model2 = model(x_train, x_test, y_train, y_test, RandomForestClassifier(), "Random_Forrest")

    models = [model1, model2]
    
    search_params = [None, random_grid]
    i = 0
    for model in models:
        model.train_and_fit(search_params[i])
        i += 1
        model.print_metrics()

        model.plot_confusion_matrix()

    

