import pandas as pd
import os

# load data from files

os.chdir('/Users/eugenekim/')
tree = pd.read_csv('Downloads/new_york_tree_census_2015.csv')


# extract the necessary columns from the data set

def column_extracter(data):
    important_col = []
    for i in [0, 3, 5, 7, 9, 10, 11, 12, 13, 14, 29, 37, 38]:
        important_col = important_col + [data.columns[i]]
    return data.loc[:, important_col]

tree = column_extracter(tree)

# choose the desired location in New York (Bronx for this project)

tree = tree[tree['boroname'] == 'Bronx'].drop(columns=['boroname'])
tree = tree.set_index('tree_id')
