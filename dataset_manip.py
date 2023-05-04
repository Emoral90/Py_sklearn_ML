from sklearn.datasets import load_iris
from sklearn import datasets
from numpy import ndarray

'''
There are two main pieces of a data set. 1) Features and the 2) response

1) Features are the variables of data. They may also be known as predictors, inputs or attributes. Regardless, there are 2 attributes of each feature
    1.1) Feature matrix: A collection of features
    1.2) Feature names: A list of all the feature names

2) The response is the output variable that relies on the features. They may also be known as the target, label, or output. Regardless, there are 2 attributes of the response
    2.1) Response vector: A response column, typcially only one
    2.2) Target names: Possible values taken by a response vector 
'''

# Load and save the dataset to a variable
iris = datasets.load_iris()
# Save its data, target, feature and target names to separte variables
data = iris.data()
target = iris.target()
# TODO Find methods on how to return feature and target names
feat_names = iris
targ_names = iris
# Display results
print(f"Feature names: {feat_names}")
print(f"Target names: {targ_names}")
print("First 10 rows of iris data:\n", data[:10])