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
from sklearn.model_selection import cross_val_score, KFold
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


#Apply Smote to training set data
#PARAM x_train: training data features
#PARAM y_train: training data targets
#Returns x and y train after having applied smote
def unbalance_dataset(x_train, y_train):

    sm = SMOTE()
    x_res, y_res = sm.fit_resample(x_train, y_train)

    return x_res, y_res

'''
Generate results from applying cross-validation from 2-K folds
args:
X: data features
Y: data targets
K_Folds: max number of folds
Model: Classifier used 
'''
def apply_cross_validation(X, Y, K_Folds, Model):

    
    fold_results = []
    #Create List of K values to specify x position of box plots
    x = []
    for k in range(2, K_Folds+1):
        x.append(k)
        folds = KFold(n_splits=K_Folds, random_state=42, shuffle=True)
        k_scores = cross_val_score(Model, X, Y, cv=folds)
        fold_results.append(k_scores)
        print("Cross Validation Accuracy of {0}".format(k_scores.mean()))
    
    
    plot = plt.boxplot(fold_results, 'red', positions=x)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(plot[item], color='purple')

    #plt.setp(plot["boxes"], facecolor='purple')
    plt.show()

'''
Utility function to format printing certain params
'''
def format_print(text):

    print("################")
    print("## {}".format(text))
    print("################")

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
            'LEARNING_RATE': 0.25,
            'DATASPLIT': 0.33,
            'PROBLEM': "Classification",
            'DATA_SPLIT': 0.1,
            'SMOTE': True
        }
    
    

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

    # Define Classifiers being used

    classifiers = [DecisionTreeClassifier(), RandomForestClassifier()]

    ##########################
    #   Cross Validation Code
    ##########################
    
    format_print("Cross Validation")
    for classifier in classifiers:
        apply_cross_validation(features, targets, 5, classifier)

    ##########################
    #   Train Test Split approach
    ##########################
    format_print("Train Test Split")
    x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=config['DATA_SPLIT'], random_state=42)

    if config['SMOTE'] == True:
        format_print("Applying Smote")
        x_train, y_train = unbalance_dataset(x_train, y_train)

    model1 = model(x_train, x_test, y_train, y_test, DecisionTreeClassifier(), "Decision_Tree")
    model2 = model(x_train, x_test, y_train, y_test, RandomForestClassifier(), "Random_Forrest")

    models = [model1, model2]
    d_tree_params =  {
        'max_depth': [ 5, 10, 20, 30, 50, 100],
        "max_features": [5],
        'min_samples_leaf': [5, 10, 20, 50, 100, 200],
        'criterion': ["gini", "entropy"]
    }

    search_params = [None, None]
    i = 0
    #Iterate through specified classifiers
    for model in models:
        #Train Model with optional random search param list
        model.train_and_fit(search_params[i])
        i += 1
        #Display results
        model.print_metrics()
        #Plot Confusion matrix
        model.plot_confusion_matrix()

    

