"""
Author: Shivam Jindal
Date: 6/13/18

- Uses Titanic Dataset to create a model that predicts likely survivors
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, ShuffleSplit, cross_val_score

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
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

    # initialize classifier and fit
    """
     Random Forest Classifier works by making multiple decision trees
     and merging them together
    """
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)


    # Get Accuracy
    """
     Shuffle Split - yields indices for training and testing
     cross_val_score - Uses standard k-fold cross validatoin
                     - splits data into folds and then trains/evaluates 
                        on each fold. You then get the average and 
                        std Dev of scores
    """
    shuffle_validator = ShuffleSplit(len(X), n_iter=20, test_size=0.2, random_state=42)
    scores = cross_val_score(clf, X, y, cv=shuffle_validator)
    print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std()))









