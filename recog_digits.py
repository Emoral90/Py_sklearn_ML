import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

# Load and save the sample dataset to a variable
digits = datasets.load_digits()

_, ax = plt.subplot(nrows=1, ncols=4, figsize=(10,3)) # Define the number of rows, columns, and size of the subplot 

for ax, image, label in zip(ax, digits.images, digits.target):
    pass