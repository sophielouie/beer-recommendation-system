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
import plotly.express as px
import jenkspy
from node2vec import Node2Vec as n2v
import node2vec

def clean_beer_reviews():
    # storing beer review dataset
    beer_reviews = pd.read_csv("beer_reviews.csv", encoding="utf-8")
    # creating a unique identifier for each beer using brewery name and beer name
    beer_reviews['Unique Beer Name'] = beer_reviews['brewery_name'] + ' ' + beer_reviews['beer_name']
    # storing beer profile dataset
    beer_profile = pd.read_csv("beer_profile_and_ratings.csv", encoding="utf-8")
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


def create_train_test_split(frac_rem=1):
    #frac_rem: fraction of each user in test set to remain in train set
    # train_set is copy of df --> df has review_profilename and Unique Beer Name as indicies
    # we are preserving user information by reseting indicies, so when we .loc[] we still have access to user and beer info
    train_set = df.copy().reset_index()
    test_parameters = []
    test_set = pd.DataFrame(columns = train_set.columns)


    for user in test_users: 
        # all reviews for a the test user
        user_reviews = train_set.loc[train_set.review_profilename == user]

        # sorted reviews
        user_reviews = user_reviews.sort_values(by = 'review_overall', ascending = False)

        # store highest reviewed beer
        highest_reviewed_beer = user_reviews.iloc[0]
        test_parameters.append((user, highest_reviewed_beer["Beer Name"]))

        # calculating the last index to remove from the train set
        last_idx = int((len(user_reviews) - 1) * frac_rem)

        # concatenate the removed user-beer pairs to test set
        test_set = pd.concat([test_set, user_reviews.iloc[1:last_idx]])

        # remove all beers from training set, add back in highest_reviewed_beer
        train_set.drop(user_reviews.iloc[1:last_idx].index, axis = 0, inplace = True)

    return train_set, test_set, test_parameters


#iterate through each column; in each column we want to qcut on that column by x bins
#Make a new column that describes which buckets the beer falls into for each descriptor
#We should have a DF where each beer has a column describing the profile buckets it is in 
#These will be used make links between beers
def bucket_me(x, df):
    # class boundaries dictionary
    boundaries = {}
    for col in df.drop(columns=['Style']).columns:
        # storing boundaries
        boundaries[col] = jenkspy.jenks_breaks(bp[col], n_classes=x)     
        df[col] = df[col].apply(lambda z: categorize(z, boundaries[col], col))
    return df


def categorize(val, boundaries, attr):
    for bound in range(len(boundaries)):
        if boundaries[bound] <= val and val < boundaries[bound + 1] + 1:
            return attr + str(bound + 1)    



def str_to_list(input):
    return input.split(',')



def beer_comparer(comp_list, beer_list, shared_att):
  # compares two lists and returns True if number of shared values is greater than or equal to shared_att
  return len([i for i, j in zip(comp_list, beer_list) if i == j]) >= shared_att


def generate_network(df, edge_col = "buckets", shared_att = 3):
    edge_dct = {}

    # iterating for each unique beer in the df
    for beer in list(df.index):
        # get  "all topic" of the beer
        beer_topics = df.loc[beer][edge_col]

        # creating a list of all the beers that are not the current one and share X attributes
        edge_df = df[(df.index != beer) & (df[edge_col].apply(beer_comparer, args = (beer_topics, shared_att, )))]
        edge_dct[beer] = edge_df.index
    
    # create nx network
    g = nx.Graph(edge_dct, create_using = nx.MultiGraph)
    return g


def predict_links(g, df, beer_name, num_rec):
    #dataframe with just row of given beer name
    this_beer = df[df.index == beer_name]

    #getting beers which are not already linked to the given beer
    
    all_nodes = g.nodes()
    #list of all beer names that are not adjacent to the beer
    all_other_nodes = [n for n in all_nodes if n not in list(g.adj[beer_name]) + [beer_name]]
    #DataFrame that contains non-adjacent nodes
    other_nodes = df[df.index.isin(all_other_nodes)]
    #find the cosine similarity between the given beer and all beers that are not already neighbors
    similar = dict()
    for beer in other_nodes.iterrows():
        similar[beer[0]] = (1 - spatial.distance.cosine(beer[1], np.array(this_beer)))
    #sort the dictionary by highest cosine similarity
    similar = pd.DataFrame(similar.items(), columns = ['beer', 'cos sim'])
    sorted_sim = similar.sort_values(by = 'cos sim', ascending = False)
    return sorted_sim['beer'].iloc[0:num_rec]



def node2vec(beer, n_recs):
    # Loading in pkl file storing graph
    filehandler = open('jenkspy_beer_network.pkl', 'rb') 
    g = pickle.load(filehandler)
    # Reading in the embedding dataframe
    emb_df = pd.read_csv('embedding_df.csv')
    # Set embedding index to beer names
    emb_df.set_index(['Unnamed: 0'], inplace=True)
    
    return predict_links(g, emb_df, beer, n_recs)


# Using the beer_profile dataframe, we want to compare 
def compile_rec_list_n2v(n_recs):
    
    rec_dict = {}
    # iterate through test users, get X recommendations for their only beer review, compare recommendations to actual ratings stored in test_set using DCG
    for user, beer in test_parameters:
        recommendations = node2vec(beer, n_recs)
        rec_dict[user] = list(recommendations)

    return rec_dict



def calc_score(metric, rec_dict, n_recs, test):
    # test is a dataframe consisting of the user reviews to be used to determine relevancy
    score = 0
    num_rel = 0
    # list that contains average precision per user
    ap = []
    # results list
    results = []
    for user, recs in rec_dict.items():
        index = 1
        num_rel = 0
        score = 0
        #dictionary containing user results
        user_metrics = {'User': user}
        for rec in recs:

            # we need to include Raise And Catch Exception when beer is not in test set
            row = test.loc[(test.review_profilename == user) & (test["Unique Beer Name"] == rec)]

            # if rec in test_set and above threshold
            if len(row) > 0:
                if metric == 'Tried':
                    score += 1
                elif row["review_overall"].iloc[0] >= THRESHOLD:
                    num_rel += 1

                    if metric == "CG":
                        score += 1

                    elif metric in ["DCG", 'NDCG']:
                        score += (1 / math.log2(index + 1))

                    elif metric == "MAP@K":
                        score += num_rel / index

                    else:
                        raise Exception("Metric type provided is not valid.")
            index += 1
        if metric == 'NDCG':
            ideal = 0
            for num in range(num_rel):
                # num starts at 0, so we add two to mimic the starting index of 1
                ideal += (1) / math.log2(num + 2)

            if ideal == 0:
                score = 0
            else:
                score = score / ideal
            
        
        if metric != 'MAP@K':
            user_metrics[metric] = np.round(score, 4)
            print(f"{user} : {np.round(score, 4)}")
        elif num_rel < 1:
            ap.append(0)
        else:
            user_metrics['AP@K'] = np.round(score / num_rel, 4)
            ap.append(score / num_rel)
        results.append(user_metrics)
    if metric == "MAP@K":
        score = np.mean(np.array(ap))
        print(np.round(score, 4))
    return pd.DataFrame(results).set_index('User')



def experiment_sharedatt(n_recs, sa):
    g = generate_network(new_bp, shared_att = sa)
    embedding = n2v(g, dimensions = 16)
    model = embedding.fit(window = 1, min_count = 1, batch_words = 4)
    emb_df = (pd.DataFrame([model.wv.get_vector(str(n)) for n in g.nodes()],
                       index = g.nodes))
    print(nx.info(g))
    nx.draw(g)
    return g, emb_df



def experiment_node2vec(network, emb, beer, n_recs):
    g = network
    emb_df = emb
    
    return predict_links(g, emb_df, beer, n_recs)



def experiment_compile_rec_list_n2v(n_recs, network, emb):
    
    rec_dict = {}
    # iterate through test users, get X recommendations for their only beer review, compare recommendations to actual ratings stored in test_set using DCG
    for user, beer in test_parameters[:5]:
        recommendations = experiment_node2vec(network, emb, beer, n_recs)
        rec_dict[user] = list(recommendations)

    return rec_dict