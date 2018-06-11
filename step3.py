import pandas as pd
from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse import linalg

# Resources : http://sdsawtelle.github.io/blog/output/week9-recommender-andrew-ng-machine-learning-with-python.html

def preProcessing():
    global training_set, training_coo

    print ("Start: Pre-processing")

    training_set = pd.read_csv("data/step0_training.csv")

    # Create the Training set Matrix
    users_row = training_set['user_id'] - 1
    anime_movie_col = training_set['anime_id'] - 1
    users_anime_data = training_set['rating'].astype(float)

    training_coo = coo_matrix((users_anime_data, (users_row, anime_movie_col)))

    print ("End: pre-processing")

def SVDPQ():
    global P, Q
    print ("Start: Perform SVD")

    # Compute SVD Q and part of P
    _k = 60
    print ("k:" + str(_k))
    P, s, Qt = linalg.svds(training_coo, k=_k)

    # Compute P
    Q = np.transpose(np.dot(np.diag(s), Qt))


    # Dimensions
    # Q = movies * factors
    # P = user * factors
    
    Q = np.random.rand(training_set['anime_id'].max(),_k)
    P = np.random.rand(training_set['user_id'].max(),_k)

    print ("End: Perform SVD")


def performGradientDescent():
    print ("Start: Perform Gradient Descent")
    # Debug Notes SVD gives zero for all ratings
    # Parameters for training -----------------------------------
    learning_rate = 0.01
    reg_rate = 0.001
    # ------------------------------------------------------------
    print("learning_rate:" + str(learning_rate) + " reg_rate:" + str(reg_rate))
    
    error_list = []

    for x in range(0, 100):
        RMSE = trainingError()

        # break if the error is gaining
        error_list.append(RMSE)

        # break the loop if you are increasing the value 
        # break the loop if you hav ereached a constant
        # Print the output
        print ("Iter:" + str(x) + " Error:" + str(RMSE))
        CalcTestingError() 
    
        if len(error_list) > 1:
            if (abs(error_list[-1] - error_list[-2]) < 0.005) or error_list[-1] > error_list[-2]:
                break

        # Update weights (P,Q)
        for index, row in training_set.iterrows():
            current_user = row.get('user_id') - 1
            current_movie = row.get('anime_id') - 1

            # Add rating to the current prediction
            current_rating = row.get('rating')

            # Get Error of the prediction
            # error = rating - prediction
            error = current_rating - np.dot(P[current_user, :], np.transpose(Q[current_movie, :]))

            # Calculate P
            P[current_user, :] = P[current_user, :] + (learning_rate * (error * Q[current_movie, :]) - (reg_rate * P[current_user, :]))

            # Calculate Q
            Q[current_movie, :] = Q[current_movie, :] + (learning_rate * (error * P[current_user, :]) - (reg_rate * Q[current_movie, :]))

    print("End: Perform Gradient Descent")


def CalcTestingError():
    # Load the testing set
    testing_set = pd.read_csv("data/step0_test.csv")

    error_prediction = pd.DataFrame()
    ratings_list = []
    prediction_list = []

    # Predict Values
    for index, row in testing_set.iterrows():
        current_user = row.get('user_id') - 1
        current_movie = row.get('anime_id') - 1

        # Add rating to the current prediction
        current_rating = row.get('rating')
        ratings_list.append(current_rating)

        prediction = np.dot(P[current_user, :], np.transpose(Q[current_movie, :]))
        prediction_list.append(prediction)

        # Get Error
    error_prediction['rating'] = pd.Series(ratings_list).values
    error_prediction['prediction'] = pd.Series(prediction_list).values

    error = ((error_prediction.rating - error_prediction.prediction) ** 2).mean() ** .5

    print ("Testing Error:" + str(error))


def trainingError():
    error_prediction = pd.DataFrame()
    ratings_list = []
    prediction_list = []

    for index, row in training_set.iterrows():
        current_user = row.get('user_id') - 1
        current_movie = row.get('anime_id') - 1

        # Add rating to the current prediction
        current_rating = row.get('rating')
        ratings_list.append(current_rating)

        prediction = np.dot(P[current_user, :], np.transpose(Q[current_movie, :]))
        prediction_list.append(prediction)

    # Get Error
    error_prediction['rating'] = pd.Series(ratings_list).values
    error_prediction['prediction'] = pd.Series(prediction_list).values

    error = ((error_prediction.rating - error_prediction.prediction) ** 2).mean() ** .5
    return error

if __name__ == '__main__':
    preProcessing()
    SVDPQ()
    performGradientDescent()
    CalcTestingError()
