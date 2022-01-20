import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# load data from files
os.chdir('/Users/eugenekim')
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

# stratify and split the data using the user_type category
X_train, X_test, y_train, y_test = train_test_split(tree, tree['health'],
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=tree['user_type'])

# plot the distribution
X_train['user_type'].hist()

# display map of bronx with the tree data
X_train.plot(kind='scatter', x='longitude', y='latitude',
             alpha=0.05, figsize=(10, 7))

# plot distribution of each category of health
plt.figure(figsize=(10, 5))
plt.bar(['Fair', 'Good', 'Poor', 'Dead'], [8702, 53289, 2476, 3695])

# before and after plot using outlier transformer on data
outlier_pre = pd.DataFrame(X_train['tree_dbh'].copy())
outlier_pre.hist(bins=25, figsize=(5, 5))
outlier_post = outlier_remover.fit_transform(outlier_pre)
outlier_post.hist(bins=25, figsize=(5, 5))

# create heatmap of results based on model selection
target_names = ['Dead', 'Poor', 'Fair', 'Good']
clf_report = classification_report(train_label, predictions,
                                   target_names=target_names,
                                   zero_division=0,
                                   output_dict=True)
sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True, cmap="Blues")

# create bar graph of results
report_data = []
for label, metrics in clf_report.items():
    if label != 'accuracy':
        metrics['label'] = label
        report_data.append(metrics)
report_df = pd.DataFrame(report_data, columns=['label', 'precision', 'recall',
                                               'f1-score', 'support'])
report_df.plot(y=['precision', 'recall', 'f1-score'], x='label', kind='bar')
