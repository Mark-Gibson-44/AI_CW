from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import balanced_accuracy_score, accuracy_score, average_precision_score

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Class Encapsulating classifier functions
class model:

    '''
    Args:
    x_train: training features
    x_test: testing features
    y_train: training targets
    y_test:  testing targets
    model: classifier to be trained
    model_name: string representing model name
    '''
    def __init__(self, x_train, x_test, y_train, y_test, model, model_name):

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.model_name = model_name
    

    def train_and_fit(self, grid=None):
        
        #Only apply RandomSearch if pass as relevant argument
        if grid is not None:
            
            self.model = RandomizedSearchCV(estimator = self.model, param_distributions = grid,
               n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)

            self.model.fit(self.x_train, self.y_train)

            self.predictions = self.model.predict(self.x_test)

            return
        # Fit features to targets
        self.model.fit(self.x_train, self.y_train)
        #save predictions when applying trained model to test data
        self.predictions = self.model.predict(self.x_test)


    '''
    Function to display multiple metrics after the model had been trained
    '''
    def print_metrics(self):
        print(self.model_name)

        print("METRICS")

        print("Accuracy: {}".format(accuracy_score(self.predictions, self.y_test)))
        print("Balanced accuracy: {}".format(balanced_accuracy_score(self.predictions, self.y_test)))

    '''
    Function to display and create plot representing precision and recall of trained model
    Args:
    f_name: optional file name if wanting to write plot to a file
    '''
    def plot_precision_recall(self, f_name=None):

        PrecisionRecallDisplay.from_predictions(self.y_test, self.predictions)
        
        #If no Filename Provided, Do not create PNG
        if f_name is not None:
            plt.savefig(f_name)
        plt.show()

    '''
    Function to display and create confusion matrix of trained model
    Args:
    f_name: optional file name if wanting to write plot to a file
    '''
    def plot_confusion_matrix(self, f_name=None):
        
        cm = confusion_matrix(self.y_test, self.predictions, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot()
        if f_name is not None:
            plt.savefig(f_name)
        plt.show()




