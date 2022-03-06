import pandas as pd
import numpy as np

'''
Function to read in data
args:
f_name: CSV file name for dataset

returns: Pandas Dataframe generated from file
'''
def read_data(f_name):
    

    return pd.read_csv(f_name)



'''
Converts coordinates in the form (x, y) into a singular value

Width is defaulted to 8 for purposes of a chess board
'''
def calculate_1d_position(x, y, width=8):
    assert x > -1 and y > -1

    return ((y-1) * 8) + (x - 1)

'''
Convert categorical piece positions into numeric equivalents
'''
def normalise_x_y_coordinates(df, x_col_name, y_col_name):
    #Dictionary of conversions
    x_pos = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': 4,
        'e': 5,
        'f': 6,
        'g': 7,
        'h': 8
    }
    
    
    encoded_elements = []
    for i in range(len(df)):
        x = x_pos[df[x_col_name][i]]
        y = df[y_col_name][i]
        
        encoded_elements.append(calculate_1d_position(x, y))

    return encoded_elements


'''
calculate distance between two pieces within the data
'''
def distance(df, r1, r2):
    dist = []

    for i in range(len(df)):
        dist.append(abs(df[r1][i] - df[r2][i]))

    return dist


'''
Generate a numeric column from a categorical column
'''
def label_to_number(data, column):
    count = 0
    #Iterate through unique elements and replace all instances of that element 
    for label in data[column].unique():
        data[column] = data[column].replace(label, count)
        count += 1
    return data[column]