"""Sim module."""


import pandas as pd
import numpy as np
import math
import Levenshtein
from nltk.util import ngrams
from nltk.metrics.distance import edit_distance
from scipy.spatial.distance import cityblock, euclidean
from datetime import datetime, timedelta


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
def dis_hamming(df, query, feature):
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate Hamming distance between query value and each value in the feature column
    hamming_distances = df[feature].apply(lambda x: hamming_distance(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(hamming_distances, columns=[feature])

    return df[feature]

# Define the calculate_text_similarity_levenshtein function
def dis_levenshtein(df, query, feature):
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

# Define the hamming_similarity function
def normalized_hamming_distance(str1, str2):
  """Calculates the normalized Hamming distance between two strings."""
  ham_dist = hamming_distance(str1, str2)
  # Get the maximum length of the strings
  max_len = max(len(str1), len(str2))
  # Normalize by the maximum length
  ham_sim = 1 - (ham_dist / max_len)
  return ham_sim

# Define the calculate_text_similarity_humming function # change the distance
def sim_hamming(df, query, feature):
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate Hamming distance between query value and each value in the feature column
    hamming_similarities = df[feature].apply(lambda x: normalized_hamming_distance(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(hamming_similarities, columns=[feature])

    return df[feature]

def normalized_levenshtein_distance(str1, str2):
  """Calculates the normalized Levenshtein distance between two strings."""
  # Get the Levenshtein distance
  lev_distance = levenshtein_distance(str1, str2)
  # Get the length of the longer string
  max_len = max(len(str1), len(str2))
  # Normalize by the maximum length
  lev_sim = 1 - (lev_distance / max_len)
  return lev_sim

# Define the calculate_text_similarity_levenshtein function
def sim_levenshtein(df, query, feature):
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate Levenshtein distance between query value and each value in the feature column
    levenshtein_similarities = df[feature].apply(lambda x: normalized_levenshtein_distance(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(levenshtein_similarities, columns=[feature])

    return df[feature]

def level_similarity(level1, level2):
  """
  Calculates the similarity score between two levels (small, medium, large).

  Args:
      level1: The first level string (e.g., "small").
      level2: The second level string (e.g., "medium").

  Returns:
      A similarity score between 0 and 1.
  """
  options = ["small", "medium", "large"]
  distance = abs(options.index(level1) - options.index(level2))
  max_distance = len(options) - 1
  if level1 == level2:
    return 1
  elif distance == 1:
    return 0.5
  else:
    return 0

# Define the calculate_text_similarity_level function
def sim_level(df, query, feature):
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate n-gram similarity between query value and each value in the feature column
    level_similarities = df[feature].apply(lambda x: level_similarity(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(level_similarities, columns=[feature])

    return df[feature]

def similarity_time(user_time, opening_time, closing_time):
    user_time = datetime.strptime(user_time, "%H:%M")
    opening_time = datetime.strptime(opening_time, "%H:%M")
    closing_time = datetime.strptime(closing_time, "%H:%M")

    if opening_time <= user_time <= closing_time:
        time_difference = closing_time - user_time
        hours_difference = time_difference.total_seconds() / 3600  # Convert to hours
        if hours_difference >= 4:
            return 1
        elif hours_difference > 0:
            return 0.5
        else:
            return 0
    else:
        return 0

def calculate_st_difference(df, user_time, opening_feature, closing_feature):
    # Calculate similarity scores for each row in df based on user's time
    similarity_scores = df.apply(lambda row: similarity_time(user_time, row[opening_feature], row[closing_feature]), axis=1)

    # Add similarity scores as a new column in df
    df['time'] = similarity_scores

    return df['time']


def sim_weighted(df, query, weights):
  similarity_results = [similarity_functions[feature](df, query, feature) for feature in df.columns]
  result = pd.concat(similarity_results, axis=1)
  weighted_df = result * pd.Series(weights)
  weighted_df['weighted_total'] = weighted_df.sum(axis=1)

  return weighted_df

def case_higher_than_query_similarity(query, case):
    if case > query:
        return 0.0
    else:
        return 1.0
#If the case value is higher than the query value, the similarity will always be 0.0

def query_higher_than_case_similarity(query, case):
    if query > case:
        return 0.0
    else:
        return 1.0
#If the query value is higher than the case value, the similarity will always be 0.0

def query_exact_match(query, case):
    if query == case:
        return 1.0
    else:
        return 0.0
#If the query value is different from the case value, the similarity will always be 0.0

def check_string_em(str1, str2):
    """
    Check if two strings are an exact match (case-insensitive) and return similarity score.

    Args:
        str1: The first string.
        str2: The second string.

    Returns:
        float: 1.0 if the strings are an exact match (case-insensitive), 0.0 otherwise.
    """
    if str1.strip().lower() == str2.strip().lower():
        return 1.0
    else:
        return 0.0

# Define the calculate_text_similarity_exact_match function
def sim_stringEM(df, query, feature):
    """
    Checks if two strings are an exact match (case-insensitive) and returns the similarity scores.

    Args:
        df: The case charactrization.
        query: The query being checked.
        feature: The specific feature.

    Returns:
        A column containing the similarity scores.
    """
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate Levenshtein distance between query value and each value in the feature column
    sem_similarities = df[feature].apply(lambda x: check_string_em(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(sem_similarities, columns=[feature])

    return df[feature]

# Define the case_higher function
def sim_CaseHigher(df, query, feature):
    """
    If the case value is higher than the query value, the similarity will always be 0.0.

    Args:
        df: The case charactrization.
        query: The query being checked.
        feature: The specific feature.

    Returns:
        A column containing the similarity scores.
    """
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate "case higher" distance between query value and each value in the feature column
    ch_distances = df[feature].apply(lambda x: case_higher_than_query_similarity(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(ch_distances, columns=[feature])

    return df[feature]


# Define the calculate_query_higher_similarity function
def sim_QueryHigher(df, query, feature):
    """
    If the query value is higher than the case value, the similarity will always be 0.0.

    Args:
        df: The case charactrization.
        query: The query being checked.
        feature: The specific feature.

    Returns:
        A column containing the similarities.
    """
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate "query higher" distance between query value and each value in the feature column
    qh_distances = df[feature].apply(lambda x: query_higher_than_case_similarity(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(qh_distances, columns=[feature])

    return df[feature]

# Define the exact match similarity function
def sim_numEM(df, query, feature):
    """
    Check if the query and the case are an exact match. (Only works for numeric data type)

    Args:
        df: The case charactrization.
        query: The query being checked.
        feature: The specific feature.

    Returns:
        A column with the similarities.
    """
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate the "exact match" distance between query value and each value in the feature column
    em_distances = df[feature].apply(lambda x: query_exact_match(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(em_distances, columns=[feature])

    return df[feature]

def log_similarity(query, case):
    """
    Calculate similarity score based on the log values (base 10) of two numeric values.

    Args:
        query: The query numeric value.
        case: The case numeric value.

    Returns:
        float: Similarity score between 0 and 1.
    """
    # Convert values to their logarithmic values (base 10)
    log_query = math.log10(query)
    log_case = math.log10(case)

    # Calculate the absolute difference between the log values
    distance = abs(log_query - log_case)

    # Convert the distance to a similarity score between 0 and 1
    # Here we assume a maximum possible distance for normalization, for instance, log10(max_value) - log10(min_value)
    # If you know the expected range of your values, you can use that for better normalization
    max_distance = math.log10(10**6)  # Example max range for normalization
    similarity_score = max(0, 1 - distance / max_distance)

    return similarity_score

# Define the log similarity function
def sim_logDifference(df, query, feature):
    """
    Calculate similarity score based on the log values (base 10) of two numeric values.

    """
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate the "exact match" distance between query value and each value in the feature column
    log_distances = df[feature].apply(lambda x: log_similarity(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(log_distances, columns=[feature])

    return df[feature]
