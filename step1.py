import pandas as pd
import multiprocessing
import numpy as np

"""
Step 1: Baseline
    1. Get the mean
    2. Rating deviation of the test set
    3. Avg. movie rating fot the test set
    4. Predict the test rating
    4. Calc Error
"""


def baselineMultiProcess(arguments):
    start_value, stop_value, mean_total, process_dataframes, baseline_list = arguments

    test_set = process_dataframes.test_set
    avg_users = process_dataframes.avg_users
    avg_movies = process_dataframes.avg_movies

    mean_total = mean_total.value

    baseline_prediction = []

    for value in range(start_value, stop_value):

        real_user = test_set.iloc[value]['user_id']
        real_movie = test_set.iloc[value]['anime_id']
        real_rating = test_set.iloc[value]['rating']

        predict = mean_total

        # add the user deviation
        predict_user = avg_users[avg_users.user_id == real_user]
        if not predict_user.empty:
            predict += predict_user.iloc[0]['mean'] - mean_total

        # add the movie deviation
        predict_movie = avg_movies[avg_movies.anime_id == real_movie]
        if not predict_movie.empty:
            predict += predict_movie.iloc[0]['mean'] - mean_total

        baseline_prediction.append({'user_id': real_user, 'anime_id': real_movie,
                                    'rating': real_rating, 'baseline': predict})

    baseline_list.extend(baseline_prediction)



def baseline():
    print ("Calculating Baseline")

    # The mean rating of every value in the training set
    mean_total = training_set['rating'].mean()

    # Avg rating of each user
    avg_users = training_set.groupby('user_id')['rating'].agg([pd.np.mean])
    avg_users = avg_users.reset_index()

    # Avg rating of each movie
    avg_movies = training_set.groupby('anime_id')['rating'].agg([pd.np.mean])
    avg_movies = avg_movies.reset_index()

    # multi-processing
    # The manager has all the dataframes
    manager = multiprocessing.Manager()
    process_dataframes = manager.Namespace()
    process_dataframes.test_set = test_set
    process_dataframes.avg_users = avg_users
    process_dataframes.avg_movies = avg_movies

    process_mean_total = manager.Value('mean_total', mean_total)
    baseline_list = manager.list()

    # Separate the Test set into cpu_count() -1 chucks
    cpus = multiprocessing.cpu_count() - 2
    num_items = len(test_set.index)
    num_process = int(num_items / cpus) + 1
    current_process = 1
    items_sent = []

    while (current_process - 1) * num_process < num_items:

        if (current_process * num_process) > num_items:
            items_sent.append((((current_process - 1) * num_process), num_items,
                               process_mean_total, process_dataframes, baseline_list))
        else:
            items_sent.append((((current_process - 1) * num_process), (current_process * num_process),
                               process_mean_total, process_dataframes, baseline_list))

        current_process += 1

    print ("These are " + str(cpus) + " processes begin started")

    pool = multiprocessing.Pool(processes = int(cpus))
    pool.map(baselineMultiProcess, items_sent)
    pool.close()
    pool.join()

    # Save it to a data frame and calculate the error
    final_test_set = pd.DataFrame.from_records(baseline_list)
    final_test_set.to_csv("data/step1.csv")

    print ("The Baseline Error: " + str(((final_test_set.rating - final_test_set.baseline) ** 2).mean() ** .5))

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
    This will be better from step two. 
    -- Training set = 80%
    -- Test set = 20%    

"""

def preProcessing():
    global training_set, test_set
    print ("Starting pre-processing")

    # load the data set onto the files
    training_set = pd.read_csv("data/step0_training.csv")
    test_set = pd.read_csv("data/step0_test.csv")

    # Set a blank value from baseline
    # test_set["baseline"] = np.nan

    print ("Ending pre-processing")


if __name__ == '__main__':
    preProcessing()
    baseline()
