"""
Author: Shivam Jindal
Date: 6/13/18

- Uses Titanic Dataset to create a model that predicts likely survivors
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split, ShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    df = df.drop(['Name', 'Ticket', 'Cabin', "Embarked"], axis=1) # possible but unlikely for these to help our model
    df = df.dropna()

    # Split dataset into numerical attributes and categorical attributes
    titanic = df.drop(["Survived"], axis=1)
    y = df["Survived"]

    # change sex from text values to binary (male and female will become 0/1)
    lb = LabelBinarizer()
    titanic.Sex = lb.fit_transform(titanic.Sex)

    # Preprocess Numerical Data by Scaling
    """
     StandardScaler is a way to standardize the range of data--
     This is helpeful because it essentially normalizes the data,
     puts mean at zero, and unit variance at zero
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(titanic)

    # breaks up the data so we can train and test it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # initialize classifier and fit
    """
     Random Forest Classifier works by making multiple decision trees
     and merging them together
    """
    clf = RandomForestClassifier(bootstrap=True, max_features=2, n_estimators=20)
    clf.fit(X_train, y_train)


    # Get Accuracy on train set
    """
     Shuffle Split - yields indices for training and testing
     cross_val_score - Uses standard k-fold cross validatoin
                     - splits data into folds and then trains/evaluates 
                        on each fold. You then get the average and 
                        std Dev of scores
    """
    shuffle_validator = ShuffleSplit(20, train_size= 0.8, test_size=0.2, random_state=0)
    scores = cross_val_score(clf, X_train, y_train, cv=shuffle_validator)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))


    # This actually checks the average score on test set
    # If number here is off really off from accuracy (above),
    # then you know the model is overfitting
    predictions = clf.predict(X_test)
    print(accuracy_score(y_test, predictions))


    """
     This is a method to help figure out the best parameters for the model
     Using this, I got off the default parameters, and managed to bring average 
     up
    """
    # param_grid = [{'n_estimators': [5,10,15,20], 'max_features':[2,4,6], 'bootstrap':[True, False]}]
    # clf2 = RandomForestClassifier()
    # grid_search = GridSearchCV(clf2, param_grid, cv=5, scoring='neg_mean_squared_error')
    # grid_search.fit(X, y)
    # print(grid_search.best_params_)




