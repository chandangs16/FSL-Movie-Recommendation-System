import pandas as pd
from scipy.sparse import coo_matrix

"""
Step 0: pre-processing
Data Evaluation:
Stanford Method:
      -------------Movies-------------
      -  0  0  0  0  0  0  0  0  0  0
      -  0  0  0  0  0  0  0  0  0  0
Users -  0  0  0  0  0  0  0  0  0  0
      -  0  0  0  0  0  0  0  0  0  0
      -  0  0  0  0  0  1  1  1  1  1

      1 = test data, 0 = training data
      Users:
      80% : 0 - 58,812
      20% : 58,812 - 73,516

      Movies:
      50 %: 0- 10258
      50 %: 10259 - 34527

Our Method: 
    Randomly Pick a set of data points. 
    This will be better from step two
    Training set = 80%
    Test set = 20%       
"""
print ("starting pre-processing")
# Load the datasets
users_rating = pd.read_csv("data/movies/ratings.csv")

# Divide he training and test data
training_set = users_rating.sample(frac=0.8)
test_set = users_rating.loc[~users_rating.index.isin(training_set.index)]

# print the files onto csv
training_set.to_csv("data/step0_training.csv", index=False)
test_set.to_csv("data/step0_test.csv", index=False)

print ("pre-processing is done")


