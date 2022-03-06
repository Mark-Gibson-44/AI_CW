#!/usr/bin/env python3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
Generates a barchart of the distribution of a given categorical column
args:
df: dataframe
categorical_col_name: (string) name of column containing the categorical data the plot will create
'''
def plot_categorical_distribution(df, categorical_col_name):
    

    columns = df[categorical_col_name].value_counts()
    

    sns.barplot(columns.index, columns.values)
    plt.show()

'''
Displays Mean and Variance of a given column
args:
df: dataframe
col_name: name of column statistics will be generated for
'''
def print_column_statistics(df, col_name):
    mean = df[col_name].mean()
    var = df[col_name].var()

    print("#####################")
    print("Mean               {}".format(mean))
    print("VARIANCE:          {}".format(var))

'''
Plots most common positions for a given piece
Args:
df: dataframe with data
col1: x position of the piece
col2: y position of the piece
label_col: column representing the target(In this data is the number of turns to win)
piece_name:(string) Name of piece the function is being used on
'''
def most_common_position_labels(df, col1, col2, label_col, piece_name):

    for label in df[label_col].unique():
        print('Most common position of piece {0} for {1} is {2} {3}'.format(piece_name, label, df.loc[df[label_col] == label][col1].mode()[0], df.loc[df[label_col] ==label][col2].mode()[0]))



if __name__ == "__main__":

    data = pd.read_csv('chess.data')

    plot_categorical_distribution(data, 'sucess')

    plot_categorical_distribution(data, 'king_x')

    plot_categorical_distribution(data, 'king_y')

    
    print_column_statistics(data, 'king_y')

    most_common_position_labels(data, 'king_x', 'king_y', 'sucess', 'white_king')
