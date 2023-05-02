from sklearn import datasets

'''
Objective: To predict what kind of image will be produced given a dataset of digits. You will do so by `fitting` an `estimator`to the each individual digit (class).
'''

# Loading example datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

# Printing data and targets of a dataset
print(f"This is the data of digits:\n{digits.data}") # will NOT show full visual of data (only first and last 3 rows)
print()
print(f"This is the target of digits:\n{digits.target}")

# Data will always be presented in a 2d array, 8*8 in this case
print(f"This is the 2d array of digits:\n{digits.images[0]}") # will show the FULL visual of data

# Import the estimator class that implements support the vector classification or svm
from sklearn import svm
clf = svm.SVC(gamma=.001, C=100.) # clf meaning classifier

# Now fit the training dataset to the fit method
    # Training set referring to the data and target attributes of the loaded dataset
fitment = clf.fit(digits.data[:-1], digits.target[:-1]) # [:-1] does is makes a new array that contains everything minus the last item
print(f"This is the data now fitted:\n{fitment}")

# Now you can predict new values by using the predict() method with the data attribute of the dataset
prediction = clf.predict(digits.data[-1:])
print(f"This is the prediction:\n{prediction}")

'''
Since this dataset uses a matrix of values representing how dark each low resolution pixel is (Higher value = darker pixel), this matrix was fitted and predicted to output a 
'''