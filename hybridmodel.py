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
import pickle 


def clean_beer_reviews():
    # storing beer review dataset
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



def hybrid(user, beer, n_recs, df, svd_model):
    # beer characteristics
    values = ['review_aroma', 'review_appearance', 'review_palate', 'review_taste', 'ABV', 'Astringency', 'Body', 'Bitter', 'Sweet', 'Sour', 'Salty', 'Fruits', 'Hoppy', 'Spices', 'Malty']
    #dataframe with only the target user and a beer they have reviewed and ideally enjoy
    target_beer_vector = df.loc[(user, beer)][values]

    # list of top 50 similar beers
    similar = []
    for beer_name in df["Beer Name"].unique():
        # contains only reviews from target beer
        beer_vectors = df.loc[df["Beer Name"] == beer_name][values]
        cos_sim = list()
        # calculate cosine similarity between the target beer and all other beer reviews for each beer beer_name
        for beer_vector in np.array(beer_vectors):
            cos_sim.append(1 - spatial.distance.cosine(beer_vector, target_beer_vector))
        similar.append((beer_name, stats.mean(cos_sim)))

    # sort in decreasing order
    similar = sorted(similar, key = lambda x: x[1], reverse = True)
    sim = similar[1:50]

    # get metadata for each of 50 similar beers
    beer_idx = [i[0] for i in sim]
    
    # we are trying to make a list of the top 50 similar beers and its metadata
    # there are multiple reviews for a single beer by different users, so we will average review_overall
    # for each similar beer --> groupby and average
    beers = pd.DataFrame()
    for idx in beer_idx:
        average_rating = df[df['Beer Name'] == idx].groupby('Beer Name').mean()[['review_overall']]
        beers = pd.concat([beers, average_rating])
    
    beers = beers.reset_index()

    # create an "est" column and apply SVD.predict() to each beer
    # We use our SVD matrix to predict how likely our user would be to like each of the 50 most similar beers
    # predict using the svd_model
    beers['est'] = beers.apply(lambda x: svd_model.predict(user, x['Beer Name'], x['review_overall']).est, axis = 1)

    # sort predictions in decreasing order
    beers = beers.sort_values(by = 'est', ascending = False)

    # return top n recommendations
    return beers[:n_recs]



def hybrid_model(user, beer, n_recs, test_frac = 1):
    # Collecting and cleaning the dataset
    df = clean_beer_reviews()
    df.set_index(['review_profilename', 'Unique Beer Name'], inplace = True)
    # Load svd pkl file
    filehandler = open('svd.pkl', 'rb') 
    svd = pickle.load(filehandler)
    return hybrid(user, beer, n_recs, df, svd)
