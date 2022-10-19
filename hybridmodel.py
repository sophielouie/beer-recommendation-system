#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/sophielouie/beer-recommendation-system/blob/main/HybridModel.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[2]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Reader, Dataset, accuracy
from surprise.model_selection import train_test_split
import numpy as np
from scipy import spatial
import statistics as stats
import math
from networkx.algorithms.dag import topological_sort
import networkx as nx
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from node2vec import Node2Vec
import copy
import pickle



def print_test():
    print("test")


# # Data Collection



def clean_beer_reviews():
    # storing beer review dataset
    beer_reviews = pd.read_csv("beer_reviews.csv")
    # creating a unique identifier for each beer using brewery name and beer name
    beer_reviews['Unique Beer Name'] = beer_reviews['brewery_name'] + ' ' + beer_reviews['beer_name']
    # storing beer profile dataset
    beer_profile = pd.read_csv("beer_profile_and_ratings.csv")
    # columns to drop from beer reviews
    drop_cols = ['brewery_id', 'brewery_name',  'beer_name', 'beer_abv', 'beer_beerid']
    # dropping columns from beer reviews
    beer_reviews.drop(columns = drop_cols, inplace = True)
    # columns to drop from  beer profile
    drop_cols = ['Name', 'Style', 'Brewery', 'Description',
                 'Min IBU', 'Max IBU', 'Alcohol', 'review_aroma', 'review_appearance', 'review_palate',
                 'review_taste', 'review_overall', 'number_of_reviews']
    # dropping columns from beer profile
    beer_profile.drop(columns = drop_cols, inplace = True)
    # combining beer review and beer profile datasets to have profile of each beer attached to every review
    df_beer = pd.merge(beer_reviews, beer_profile, left_on = 'Unique Beer Name', right_on = 'Beer Name (Full)', how = 'inner')
    # isolating the numerical columns that need to be scaled
    need_scaling = df_beer.drop(columns = ['review_time', 'review_profilename', 'beer_style', 'Unique Beer Name', 'Beer Name (Full)'])
    # storing the informational portion of the dataset that does not need scaling
    informational = df_beer[['review_time', 'review_profilename', 'beer_style', 'Unique Beer Name', 'Beer Name (Full)']]
    # renaming beer name column
    informational.rename(columns = {'Beer Name (Full)': 'Beer Name'}, inplace = True)

    # scaling the data
    scaler = MinMaxScaler()
    scaler.fit(need_scaling)
    need_scaling = pd.DataFrame(scaler.transform(need_scaling), columns = need_scaling.columns)

    # recombining the informational data and scaled data
    df = pd.concat([informational, need_scaling], axis = 1)
    return df



# # Hybrid Model

# Conceptualize the hybrid model



def hybrid(user, beer, n_recs, df, svd_model):

    values = ['review_aroma',	'review_appearance',	'review_palate',	'review_taste', 'ABV', 'Astringency', 'Body', 'Bitter', 'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty']

    target_beer_vector = df.loc[(user, beer)][values]

    # list of top 50 similar beers
    similar = []
    for beer_name in df["Beer Name"].unique():
        beer_vectors = df.loc[df["Beer Name"] == beer_name][values]
        cos_sim = list()
        for beer_vector in np.array(beer_vectors):
            cos_sim.append(1 - spatial.distance.cosine(beer_vector, target_beer_vector))
        similar.append((beer_name, stats.mean(cos_sim)))

    # sort in decreasing order
    similar = sorted(similar, key = lambda x: x[1], reverse = True)
    sim = similar[1:50]

    # get metadata for each of 50 similar beers
    beer_idx = [i[0] for i in sim]

    # we are trying to make a list of the top 50 similar beers and its metadata
    # there are multiple reviews for a single beer by different users, so we will average review_overall for each similar beer --> groupby and average
    beers = pd.DataFrame()
    for idx in beer_idx:
        average_rating = df[df['Beer Name'] == idx].groupby('Beer Name').mean()[['review_overall']]
        beers = pd.concat([beers, average_rating])
  
    beers = beers.reset_index()

    # create an "est" column and apply SVD.predict() to each book
    # predict using the svd_model
    beers['est'] = beers.apply(lambda x: svd_model.predict(user, x['Beer Name'], x['review_overall']).est, axis = 1)

    # sort predictions in decreasing order
    beers = beers.sort_values(by = 'est', ascending = False)

    # return top n recommendations
    return beers[:n_recs]


# #Hybrid Testing

# We need to select users to isolate for our test set. The criteria for these users is that they should have reviewed enough beers that there is some likelihood that they would have tried something that has been recommended to them. Starting with binary cumulative gain, we will assess the performance of our recommendation system by determining if the recommended beers have been tried by the user or exceeded a threshold.

# - recommend x beers to 10 users
# - traverse the recommended beer list, see how many have been rated highly
# - compare the accuracy ratings of the three test users
# - determine threshold for positive review by finding percentiles of ratings (top 1/4 review would be considered a successful recommendation)



# ##Find threshold for what makes a good rating
THRESHOLD = 0.875




def create_train_test_split(df, test_users, frac_rem=1):
    #frac_rem: fraction of each user in test set to remain in train set
    # train_set is copy of df --> df has review_profilename and Unique Beer Name as indicies
    # we are preserving user information by reseting indicies, so when we
    # .loc[] we still have access to user and beer info
    train_set = copy.copy(df)
    train_set = train_set.reset_index()
    test_parameters = []
    test_set = pd.DataFrame(columns = train_set.columns)


    for user in test_users: 
        # all reviews for a the test user
        user_reviews = train_set.loc[train_set.review_profilename == user]

        # sorted reviews
        sorted_user_reviews = user_reviews.sort_values(by = 'review_overall', ascending = False)

        # store highest reviewed beer
        highest_reviewed_beer = sorted_user_reviews.iloc[0]
        test_parameters.append((user, highest_reviewed_beer["Beer Name"]))

        # removing highest reviewed beer from user_reviews so that it remains in the train set
        user_reviews.drop(highest_reviewed_beer.name, axis = 0, inplace = True)

        # calculating the last index to remove from the train set
        last_idx = int((len(user_reviews) - 1) * frac_rem)

        # concatenate the removed user-beer pairs to test set
        test_set = pd.concat([test_set, user_reviews.iloc[0:last_idx]])

        # remove all beers from training set
        train_set.drop(user_reviews.iloc[0:last_idx].index, axis = 0, inplace = True)

    return train_set, test_set, test_parameters




def train_svd(train_set, test_set):
    # creating and training SVD on train_set
    reader = Reader()

    # train_set = train_set.reset_index()
    # test_set = test_set.reset_index()

    test_data = Dataset.load_from_df(test_set[['review_profilename', 'Beer Name', 'review_overall']], reader)
    train_data = Dataset.load_from_df(train_set[['review_profilename', 'Beer Name', 'review_overall']], reader)

    # NEED TO TURN OUR TRAIN_SET INTO surprise TRAINSET TYPE
    surprise_train_set = train_data.build_full_trainset()

    # train
    svd = SVD()
    svd.fit(surprise_train_set)
    return svd




def hybrid_model(user, beer, n_recs, test_frac = 1):
    # Collecting and cleaning the dataset
    df = clean_beer_reviews()
    # Setting the index for easier querying
    df.set_index(['review_profilename', 'Unique Beer Name'], inplace = True)
    # Load svd pkl file
    filehandler = open('svd.pkl', 'rb') 
    svd = pickle.load(filehandler)
    return hybrid(user, beer, n_recs, df, svd)




def compile_rec_list(n_recs, test_parameters):
    rec_dict = {}
    # iterate through test users, get X recommendations for their only beer review, compare recommendations to actual ratings stored in test_set using DCG
    for user, beer in test_parameters[0:5]:
        recommendations = hybrid(user, beer, n_recs, train_set, svd)
        rec_dict[user] = list(recommendations["Beer Name"])

    return rec_dict




def calc_score(metric, rec_dict, n_recs, test):
    # test is a dataframe consisting of the user reviews to be used to determine relevancy
    score = 0
    index = 1
    num_rel = 0
    for user, recs in rec_dict.items():

        for rec in recs:
            
            # we need to include Raise And Catch Exception when beer is not in test set
            row = test.loc[(test.review_profilename == user) & (test["Unique Beer Name"] == rec)]

            # if rec in test_set and above threshold
            if len(row) > 0:
                if row["review_overall"].iloc[0] >= THRESHOLD:
                    num_rel += 1

                    if metric == "CG":
                        score += 1

                    elif metric in ["DCG", 'NDCG']:
                        score += (1 / math.log2(index + 1))

                    elif metric == "MAP":
                        score += num_rel / index

                    else:
                        raise Exception("Metric type provided is not valid.")

            if metric == 'NDCG':
                ideal = 0
                for num in range(num_rel):
                    # num starts at 0, so we add two to mimic the starting index of 1\
                    ideal += (1 / math.log2(num + 2))
    
                score = score / ideal

    if metric == "MAP":
        score = score / n_recs

    index += 1
    print(f"{user} : {score}")

