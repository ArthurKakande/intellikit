"""Sim module."""


import pandas as pd
import numpy as np
import Levenshtein
from nltk.util import ngrams
from nltk.metrics.distance import edit_distance
from scipy.spatial.distance import cityblock, euclidean


# Define the hamming_distance function
def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

# Define the Levenshtein distance function
def levenshtein_distance(str1, str2):
    return edit_distance(str1, str2)

# Define the n-gram similarity function
def ngram_similarity(str1, str2, n=2):
    grams1 = set(ngrams(str1, n))
    grams2 = set(ngrams(str2, n))
    return len(grams1.intersection(grams2)) / max(len(grams1), len(grams2))

# Define the cityblock similarity function
def cityblock_similarity(arr1, arr2):
    return cityblock(arr1, arr2)

# Define the Euclidean distance function
def euclidean_distance(arr1, arr2):
    return euclidean(arr1, arr2)

# Define the absolute distance function
def abs_difference(arr1, arr2):
    return 1- abs(arr1 - arr2)/max(arr1, arr2)

# Define the calculate_text_similarity_humming function
def sim_hamming(df, query, feature):
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate Hamming distance between query value and each value in the feature column
    hamming_distances = df[feature].apply(lambda x: hamming_distance(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(hamming_distances, columns=[feature])

    return df[feature]

# Define the calculate_text_similarity_levenshtein function
def sim_levenshtein(df, query, feature):
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate Levenshtein distance between query value and each value in the feature column
    levenshtein_distances = df[feature].apply(lambda x: levenshtein_distance(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(levenshtein_distances, columns=[feature])

    return df[feature]

# Define the calculate_text_similarity_ngram function
def sim_ngram(df, query, feature):
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate n-gram similarity between query value and each value in the feature column
    ngram_similarities = df[feature].apply(lambda x: ngram_similarity(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(ngram_similarities, columns=[feature])

    return df[feature]

# Define the calculate_numeric_similarity_cityblock function
def sim_cityblock(df, query, feature):
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate cityblock distance between query value and each value in the feature column
    cityblock_distances = df[feature].apply(lambda x: cityblock_similarity(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(cityblock_distances, columns=[feature])

    return df[feature]

# Define the calculate_numeric_similarity_euclidean function
def sim_euclidean(df, query, feature):
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate Euclidean distance between query value and each value in the feature column
    euclidean_distances = df[feature].apply(lambda x: euclidean_distance(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(euclidean_distances, columns=[feature])

    return df[feature]

# Define the calculate_numeric_similarity_abs_difference function
def sim_difference(df, query, feature):
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate Euclidean distance between query value and each value in the feature column
    euclidean_distances = df[feature].apply(lambda x: abs_difference(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(euclidean_distances, columns=[feature])

    return df[feature]

def sim_weighted(df, query, weights):
  similarity_results = [similarity_functions[feature](df, query, feature) for feature in df.columns]
  result = pd.concat(similarity_results, axis=1)
  weighted_df = result * pd.Series(weights)
  weighted_df['weighted_total'] = weighted_df.sum(axis=1)

  return weighted_df
