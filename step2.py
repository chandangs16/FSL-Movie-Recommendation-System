import pandas as pd
from scipy.sparse import coo_matrix
import sys
import multiprocessing


def preProcessing():
    """
        Load the test and training set from step two
    """
    print ("starting pre-processing")

    global training_set

    # Load the datasets
    training_set = pd.read_csv("data/step0_training.csv")

    print ("pre-processing is done")


def itemItemCollabrative():
    """
        Step 2: Item-Item Collabrative Filtering
            1. Use the baseline from previous (baseline['baseline])
            2. Calculate the mean of each movie (avg_movies)
            3. Subtract the mean from each movies (presion_ratings)
            4. Compute Cosine sim between each movie (movie_similarity)
            5. 
    """
    global baseline
    print ("Start Item-Item Collabrative Filtering")

    # Get the baseline from step1
    baseline = pd.read_csv('data/step1.csv', 
                           skipinitialspace=True, usecols=['baseline', 'rating', 'anime_id', 'user_id'])

    # The mean rating of every value in the training set
    mean_total = training_set['rating'].mean()

    # Avg rating of each user
    avg_users = training_set.groupby('user_id')['rating'].agg([pd.np.mean])
    avg_users = avg_users.reset_index()

    # Avg rating of each movie
    avg_movies = training_set.groupby('anime_id')['rating'].agg([pd.np.mean])
    avg_movies = avg_movies.reset_index()

    # Subtract the mean from each movie in the testing set
    pearson_ratings = training_set.copy()

    print ("-- Subtract the mean")
    # Take around 14-mins

    for index, row in avg_movies.iterrows():
        mean_movie = row.get('anime_id')
        mean_rating = row.get('mean')
    
        pearson_ratings.loc[pearson_ratings['anime_id'] == mean_movie, 'rating'] = pearson_ratings.loc[pearson_ratings['anime_id'] == mean_movie, 'rating'] - mean_rating

    print ("-- Compute the cosine")
    # Take around 2-mins
    # Compute the cosine matrix for using the rating
    users_col = pearson_ratings['user_id']
    anime_movie_row = pearson_ratings['anime_id']
    users_anime_data = pearson_ratings['rating']

    pearson_ratings_coo = coo_matrix((users_anime_data, (anime_movie_row, users_col)))

    # distance Top 
    distance_top = pearson_ratings_coo * pearson_ratings_coo.transpose()

    # training_coo^2 and sum each row
    distance_square = pearson_ratings_coo.multiply(pearson_ratings_coo).sum(1)
    distance_square = coo_matrix(distance_square).sqrt()

    # get the normal of bottom
    distance_normal = distance_square.dot(distance_square.transpose())

    # Get the distance
    distance_normal.data = 1/distance_normal.data

    movie_similarity = distance_top.multiply(distance_normal)
    movie_similarity = pearson_ratings_coo.tocsr() 

    print( "-- Get a N for each test entry")
                
    # multi-processing
    # The manager has all the dataframes
    manager = multiprocessing.Manager()
    process_data = manager.Namespace()
    process_data.training_set = training_set
    process_data.baseline = baseline
    process_data.avg_users = avg_users
    process_data.avg_movies = avg_movies
    process_data.movie_similarity = movie_similarity
    
    process_mean_total = manager.Value('mean_total', mean_total)
    
    # Separate the Test set into cpu_count() -1 chucks
    # https://stackoverflow.com/questions/17315737/split-a-large-pandas-dataframe/17315875
    # https://stackoverflow.com/questions/1501651/log-output-of-multiprocessing-process
    cpus = multiprocessing.cpu_count() - 2
    num_items = len(baseline.index)
    num_process = int(num_items / cpus) + 1
    current_process = 1
    items_sent = []
    items_read = []
    
    while (current_process - 1) * num_process < num_items:
        if (current_process * num_process) > num_items:
            items_sent.append((((current_process - 1) * num_process), num_items,
                               process_mean_total, process_data))
            items_read.append(str((current_process - 1) * num_process) +  "_output.csv")
        else:
            items_sent.append((((current_process - 1) * num_process), (current_process * num_process),
                               process_mean_total, process_data))
            items_read.append(str((current_process - 1) * num_process) +  "_output.csv")

        current_process += 1

    print ("These are " + str(cpus) + " processes begin started")
    pool = multiprocessing.Pool(processes = int(cpus))
    
    pool.map(stepTwoMultiProcess, items_sent)
    pool.close()
    pool.join()
    
    # Calculate the error 
    # Read file and concatenate them 
    # https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
    files_list = list()
    prediction_frame = pd.DataFrame()
    
    for file_name in items_read:
        temp_file = pd.read_csv(file_name,index_col=None, header=0)
        files_list.append(temp_file)
    
    prediction_frame = pd.concat(files_list)
    print (prediction_frame)
    
    # Calculate the error
    print ("The Baseline Error: " + str(((prediction_frame.rating - prediction_frame.prediction) ** 2).mean() ** .5))
    
    
def stepTwoMultiProcess(arguments):
    start_value, stop_value, mean_total, process_data = arguments
    
    training_set = process_data.training_set
    baseline = process_data.baseline
    avg_users = process_data.avg_users
    avg_movies = process_data.avg_movies
    movie_similarity = process_data.movie_similarity

    mean_total = mean_total.value
    
    
    
    sys.stdout = open(str(start_value) + "_output.csv", "w")
    print ("index," + "prediction," + "rating," + "user_id," + "anime_id")
    
    for value in range(start_value, stop_value):
        
        row_baseline = baseline.loc[value]
        
        # Get the values from the test set
        test_user = row_baseline.get('user_id')
        test_movie = row_baseline.get('anime_id')
        
        real_rating = row_baseline.get('rating')
        
        # Add basline to the current prediction  
        prediction = row_baseline.get('baseline')
        
        # Get the values the user has rated
        # If the user hasn't rated any other values in the training set we set the current prediction to the actually prediction
        rated_training_set = training_set[training_set.user_id == test_user].copy()
        
        if not rated_training_set.empty:
            # This mean there are no empty values in the training set
            rated_training_set = rated_training_set.reset_index(drop =True)
            
            # Add the rating values 
            def getSim(row):
                compare_movie = row.get('anime_id')
                # return the sim 
                return movie_similarity[int(test_movie),int(compare_movie)]
                
            rated_training_set['sim']  = rated_training_set.apply (lambda row: getSim (row),axis=1)
                
            # Select the best N values or less
            rated_training_set = rated_training_set.sort_values('sim', ascending = False)
            rated_training_set = rated_training_set.head(25)
            
            # compute the bottom value of the equation
            # top and bottom could be zero if there is one movie rated to compute the similiarity 
            bottom_value = rated_training_set['sim'].sum()
            
            if (int(bottom_value) != 0):
                # now compute the baseline for each values
                def getBasline(row):
                    compare_user = row.get('user_id')
                    compare_movie = row.get('anime_id')
                    
                    predict_basline = mean_total
    
                    # add the user deviation
                    predict_user = avg_users[avg_users.user_id == compare_user]
                    if not predict_user.empty:
                        predict_basline += predict_user.iloc[0]['mean'] - mean_total
    
                    # add the movie deviation
                    predict_movie = avg_movies[avg_movies.anime_id == compare_movie]
                    if not predict_movie.empty:
                        predict_basline += predict_movie.iloc[0]['mean'] - mean_total
                                                            
                    return predict_basline
                
                rated_training_set['baseline'] = rated_training_set.apply (lambda row: getBasline (row),axis=1)
                
                # Compute the calculation
                bottom_value = rated_training_set['sim'].sum()
                top_value = ((rated_training_set['rating']-rated_training_set['baseline'])* rated_training_set['sim']).sum()
                
                prediction += (top_value/bottom_value)
                
        print ( str(value) + "," +  str(prediction)  + "," + str(real_rating) + "," + str(test_user) + "," + str(test_movie))
        
        # do some stuff
        
            
if __name__ == '__main__':
    preProcessing()
    itemItemCollabrative()