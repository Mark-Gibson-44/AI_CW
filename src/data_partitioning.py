from imblearn.over_sampling import SMOTE 

def unbalance_dataset(x_train, y_train):

    sm = SMOTE()
    x_res, y_res = sm.fit_resample(x_train, y_train)

    return x_res, y_res

def _train_test_split(x_data, y_data, split_percentage=0.33):
    return train_test_split(x_data, y_data, test_size=split_percentage, random_state=42)