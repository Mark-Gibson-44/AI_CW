from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import balanced_accuracy_score, accuracy_score, average_precision_score

from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class model:

    def __init__(self, x_train, x_test, y_train, y_test, model, model_name):

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.model_name = model_name
    
    def train_and_fit(self, grid=None):

        self.model.fit(self.x_train, self.y_train)
        
        self.predictions = self.model.predict(self.x_test)



    def print_metrics(self):
        print(self.model_name)

        print("METRICS")

        print("Accuracy: {}".format(accuracy_score(self.predictions, self.y_test)))
        print("Balanced accuracy: {}".format(balanced_accuracy_score(self.predictions, self.y_test)))
        #print("Precision: {}".format(average_precision_score(self.predictions, self.y_test)))
        
    def plot_precision_recall(self, f_name=None):

        PrecisionRecallDisplay.from_predictions(self.y_test, self.predictions)
        plt.show()
        #If no Filename Provided, Do not create PNG
        if f_name is not None:
            pass

    def plot_confusion_matrix(self, f_name=None):
        
        cm = confusion_matrix(self.y_test, self.predictions, labels=self.model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.model.classes_)
        disp.plot()
        plt.show()




