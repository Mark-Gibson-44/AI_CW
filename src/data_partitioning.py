from imblearn.over_sampling import SMOTE 

'''
Apply Smote to dataset in aims of balancing the data more
args:
x_train: training features from the dataset
y_train: training targets from the dataset
'''
def unbalance_dataset(x_train, y_train):

    sm = SMOTE()
    x_res, y_res = sm.fit_resample(x_train, y_train)

    return x_res, y_res


'''
generate a train test split from dataset:
args:
x_data: features of dataset
y_data: targets of dataset
split_percentage: amount of data to be split into test data
'''
def _train_test_split(x_data, y_data, split_percentage=0.33):
    return train_test_split(x_data, y_data, test_size=split_percentage, random_state=42)