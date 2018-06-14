# Titanic Dataset

The titanic dataset is a widely known beginner dataset to help learn and get into machine learning.That's exactly why I am using this dataset -- to get into and learn more about machine learning.

## Contributors
Shivam Jindal

## Installation
* scikit-learn (0.19.0)
* pandas (0.20.3)
* numpy (1.13.3)

## How it works
Most of it is commented in the model.py file itself. It first opens the file. From there, the program drops seemingly unneccessary columns (name, Ticket, Embarked, and Cabin), and sets the rest of the data into features and labels. The features are then preprocessed such that: 
    * Sex is changed from categorical data (male or female) to numerical data (0 or 1)
    * All features are standardized
From there, the data is split into a train set and test set and validated. 

The score tends to be around 0.79 with a standard deviation of either 0.02 or 0.03

## Improvements
There are definitely stronger models that could be used instead, which I will discover down the road. Moreover, the dataset could likely be preprocessed better. There may be some other fine tuning to look into with the classifier itself. 


    


