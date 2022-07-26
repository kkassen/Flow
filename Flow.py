# Author: Kyle Arick Kassen
# Date: June 15, 2022 - June 21, 2022

# Import Statements
import pandas as pd
import IPython.display as ip
import sklearn.model_selection as ms
import sklearn.tree as dt
import sklearn.linear_model as lr
import sklearn.metrics as skm
import matplotlib.pyplot as plt

class dataRetrieval():

    # Function for retrieving data and storing in dataframe
    def load(self, filename, filetype):
        types = {'csv': pd.read_csv(filename), 'txt': pd.read_table(filename), 'json': pd.read_json}
        data = types.get(filetype)
        return data

    # Function for Ensuring data was properly retrieved
    def quality_control(self, data, option, view=False):
        options = {'data': data, 'head': data.head(), 'tail': data.tail(), 'shape': data.shape}
        x = options.get(option)
        if view and option in options.keys(): print('\n' + option.capitalize() + '-->'), print(x)
        return x

class dataExploration():

    # Function returns summary statistics for a particular dataframe
    def summary_statistics(self, data, view=False):
        stats = data.describe()
        if view: print('\nSummary Statistics -->'), print(stats)
        return stats

    # Function for reviewing data types and non-null counts for a particular dataframe
    def describe_data(self, data, view=False):
        if view: print('\nData Information -->'), data.info()
        return None

    # Function for encoding a column with 0's and 1's
    def recode(self, data, column, x):
        data[column] = data[column].apply(lambda y: 0 if y == x else 1)
        return data[column]

    # Function for calculating the occurrence as a percentage for a specified value within a specified column
    def percentages(self, data, column, value, view=False):
        counts = data[column].value_counts()[value]
        x = counts/len(data)
        if view: print('{:.2%}'.format(x))
        return x

    # Function returns the value that appears most frequently within a specified column
    def common(self, data, column, view=False):
        counts = data[column].value_counts()
        x = counts.index[0]
        if view: print(x)
        return x

    # Function returns the average value of column (c2) given a specified value for column (c1)
    def filter_average(self, data, c1, c2, value, view=False):
        x = data[data[c1] == value]
        z = sum(x[c2])/len(x)
        if view: print('{:.2f}'.format(z))
        return z

    # Function returns a correlation matrix for a specified dataframe
    def correlation(self, data, view=False, color=False):
        x = data.corr(method='pearson')
        if view: print('\nCorrelation Matrix -->'), print(x)
        if color: ip.display(x.style.background_gradient(cmap='Greens'))
        return x

class dataWrangling():

    # Function for removing non-numeric characters
    def drop_characters(self, data, column, expression):
        data[column] = data[column].str.replace(expression, '', regex=True).astype(int)
        return data[column]

    # Function for encoding a column with numeric values
    def map(self, data, dictionary):
        return data.replace(dictionary, inplace=True)

class predictiveModeling():

    # Function for partitioning the dataset into testing and training datasets
    def partition(self, data, size):
        train, test = ms.train_test_split(data, test_size=size, random_state=73)
        return train, test

    # Function for accessing training datasets
    def train_sets(self, data, features, column):
        training_features = data[features]
        training_target = data[column]
        return training_features, training_target

    # Function for accessing testing datasets
    def test_sets(self, data, features, column):
        testing_features = data[features]
        testing_target = data[column]
        return testing_features, testing_target

    # Function for machine learning model selection
    def model_selector(self, algorithm):
        algorithms = {'tree': dt.DecisionTreeClassifier(), 'lr': lr.LogisticRegression()}
        x = algorithms.get(algorithm)
        return x

    # Function for fitting a machine learning model
    def generic_model(self, M, features, target):
        model = M.fit(features, target)
        return model

    # Function for optimizing, cross-validating, and fitting a machine learning model
    def optimize(self, M, hyperparmeters, folds, metric, features, target):
        x = ms.GridSearchCV(M, hyperparmeters, cv=folds, scoring=metric)
        optimized_model = x.fit(features, target)
        return optimized_model

    # Function takes a machine learning model and produces predictions based on a specified testing dataset
    def predict(self, M, features):
        target = M.predict(features)
        return target

    # Function that outputs the results of a machine learning model
    def measurements(self, testing_target, predictions, option, M=None, X=None, Y=None, x_train=None, y_train=None):
        print('\n' + option + ' Model Results -->')
        print('Accuracy: {:.2%}'.format(skm.accuracy_score(testing_target, predictions)))
        print('Precision: {:.2%}'.format(skm.precision_score(testing_target, predictions)))
        print('Recall: {:.2%}'.format(skm.recall_score(testing_target, predictions)))
        print('F1-Score: {:.2%}'.format(skm.f1_score(testing_target, predictions)))
        print('Accuracy (Training Data): {:.2%}'.format(M.score(x_train, y_train)))
        if option == 'Optimized' and M is not None:
            x = ms.cross_val_score(M.best_estimator_, X, Y, cv=ms.StratifiedKFold(shuffle=True)).mean()
            print('Cross-Val: {:.2%}'.format(x))

    # Function that writes the test features, actuals, and predictions to a .csv file
    def write(self, x_data, y_data, predictions, c1, c2, file):
        table = pd.DataFrame(x_data, copy=True)
        table.insert(column=c1, value=y_data, loc=len(table.columns))
        table.insert(column=c2, value=predictions, loc=len(table.columns))
        return table.to_csv(file, encoding='utf-8', index=False)

class visualization():

    # Function for visualizing a decision tree
    def visualize_tree(self, M, title):
        dt.plot_tree(M)
        plt.title(title)
        return plt.show()

    # Function for visualizing a confusion matrix
    def confusion_matrix(self, target, predictions):
        cm = skm.confusion_matrix(target, predictions)
        s = skm.ConfusionMatrixDisplay(cm)
        s.plot(cmap='Greys')
        return plt.show()

    # Function returns a bar chart containing the top (n) features
    def visualize_n_features(self, M, data, n, width, height, color):
        dims = plt.figure()
        dims.set_figwidth(width)
        dims.set_figheight(height)
        barchart = (pd.Series(M.best_estimator_.feature_importances_,
                              index=data.columns).nlargest(n).plot(kind='barh', color=color))
        plt.title('Top Features', fontsize=16, pad=15)
        barchart.invert_yaxis()
        plt.xlabel('Score', fontsize=12, labelpad=20)
        plt.ylabel('Feature', fontsize=12, labelpad=20)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8, rotation=45)
        plt.tight_layout()
        return plt.show()




