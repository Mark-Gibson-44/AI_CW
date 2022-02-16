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
            'DATASPLIT': 0.33
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


    model1 = model(x_train, x_test, y_train, y_test, DecisionTreeClassifier(), "Decision_Tree")
    model2 = model(x_train, x_test, y_train, y_test, RandomForestClassifier(), "Random_Forrest")

    models = [model1, model2]

    for model in models:
        model.train_and_fit()

        model.print_metrics()

        model.plot_confusion_matrix()

    

