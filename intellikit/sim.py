"""Sim module."""

import pandas as pd
import numpy as np
import math
from nltk.util import ngrams
from scipy.spatial.distance import cityblock, euclidean
from datetime import datetime, timedelta


# Define the hamming_distance function
def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

# Define the Levenshtein distance function
def levenshtein_distance(str1, str2):
    """
    Calculate the Levenshtein distance between two strings.

    The Levenshtein distance is a measure of the difference between two strings.
    It is defined as the minimum number of single-character edits (insertions,
    deletions, or substitutions) required to change one string into the other.

    Args:
        str1 (str): The first string.
        str2 (str): The second string.

    Returns:
        int: The Levenshtein distance between the two strings.
    """
    # Initialize a matrix where dp[i][j] represents the distance between
    # the first i characters of str1 and the first j characters of str2.
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    # Set up the initial distances when one of the strings is empty.
    for i in range(len(str1) + 1):
        dp[i][0] = i
    for j in range(len(str2) + 1):
        dp[0][j] = j

    # Compute the distances.
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:  # No change needed if characters are the same.
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # Calculate costs for substitution, insertion, and deletion.
                substitution_cost = dp[i - 1][j - 1] + 1
                insertion_cost = dp[i][j - 1] + 1
                deletion_cost = dp[i - 1][j] + 1
                # Find the minimum of these three options.
                dp[i][j] = min(substitution_cost, insertion_cost, deletion_cost)

    # The bottom-right corner of the matrix contains the final Levenshtein distance.
    return dp[-1][-1]

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
    """
    Calculate the Levenshtein distance between the query value and each value in the specified feature column of a DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the feature column.
        query (DataFrame): The DataFrame containing the query value.
        feature (str): The name of the feature column.

    Returns:
        DataFrame: A DataFrame with the Levenshtein distances between the query value and each value in the feature column.
    """
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
    """
    Calculates the normalized Levenshtein distance between two strings.

    Parameters:
    str1 (str): The first string.
    str2 (str): The second string.

    Returns:
    float: The normalized Levenshtein distance between the two strings.
    """
    # Get the Levenshtein distance
    lev_distance = levenshtein_distance(str1, str2)
    # Get the length of the longer string
    max_len = max(len(str1), len(str2))
    # Normalize by the maximum length
    lev_sim = 1 - (lev_distance / max_len)
    return lev_sim

# Define the calculate_text_similarity_levenshtein function
def sim_levenshtein(df, query, feature):
    """Calculate the Levenshtein similarity between the query value and each value in the specified feature column.

    Args:
        df (DataFrame): The input DataFrame.
        query (DataFrame): The query DataFrame containing the value to compare.
        feature (str): The name of the feature column to calculate the similarity for.

    Returns:
        DataFrame: A DataFrame column containing the Levenshtein similarity values for each value in the feature column.
    """
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
      A similarity score between 0 and 1. (returns 1 if level1=level2, 0.5 if the level1 is close to level2)
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
    """
    Calculate the level similarity (small, medium, large) between the query value and each value in the specified feature column of a DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the feature column.
        query (DataFrame): The DataFrame containing the query value.
        feature (str): The name of the feature column.

    Returns:
        DataFrame: A DataFrame column with the level similarity values for each value in the feature column.
    """
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate n-gram similarity between query value and each value in the feature column
    level_similarities = df[feature].apply(lambda x: level_similarity(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(level_similarities, columns=[feature])

    return df[feature]

def similarity_time(user_time, opening_time, closing_time):
    """Calculate the similarity between the user's time and the opening and closing times.

    Args:
        user_time (str): The time entered by the user in the format "HH:MM".
        opening_time (str): The opening time in the format "HH:MM".
        closing_time (str): The closing time in the format "HH:MM".

    Returns:
        float: The similarity score between the user's time and the opening and closing times.
            - 1 if the user's time is within the opening and closing times and the difference is 4 hours or more.
            - 0.5 if the user's time is within the opening and closing times and the difference is less than 4 hours.
            - 0 if the user's time is outside the opening and closing times.
    """
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


#def sim_weighted(df, query, weights):
#  similarity_results = [similarity_functions[feature](df, query, feature) for feature in df.columns]
#  result = pd.concat(similarity_results, axis=1)
#  weighted_df = result * pd.Series(weights)
#  weighted_df['weighted_total'] = weighted_df.sum(axis=1)
#
#  return weighted_df

def case_higher_than_query_similarity(query, case):
    """Checks if a case value is higher than the query value and returns a similarity score.

    Args:
        query (_type_): The query value.
        case (_type_): The case value.

    Returns:
        float: A similarity score of 0 if the case value is higher than the query value, and 1 otherwise.
    """
    if case > query:
        return 0.0
    else:
        return 1.0

def query_higher_than_case_similarity(query, case):
    """Check if the query is higher than the case similarity.

    Args:
        query (float): The similarity score of the query.
        case (float): The similarity score of the case.

    Returns:
        float: Returns 0.0 if the query is higher than the case similarity, otherwise returns 1.0.
    """
    if query > case:
        return 0.0
    else:
        return 1.0

def query_exact_match(query, case):
    """Check if the query value is an exact match with the case value and return a similarity score.

    Args:
        query (_type_): The query value.
        case (_type_): The case value.

    Returns:
        _type_: A similarity score of 1.0 if the query value is an exact match with the case value, otherwise returns 0.0.
    """
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
    Calculate similarity score based on the log values (base 10) of a query value and a value from the dataframe.

    Args:
        df: The case charactrization.
        query: The query being checked.
        feature: The specific feature in the dataframe.

    Returns:
        Dataframe column: A column containing the similarities. 
    """
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate the "exact match" distance between query value and each value in the feature column
    log_distances = df[feature].apply(lambda x: log_similarity(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(log_distances, columns=[feature])

    return df[feature]


#Calculate the cosine similarity between two sentences
def sent_cosine_similarity(sentence1, sentence2):
    """Calculates the cosine similarity between two sentences.

    This function takes in two sentences and calculates the cosine similarity between them. 
    The cosine similarity is a measure of similarity between two non-zero vectors of an inner product space.
    It is defined as the cosine of the angle between the two vectors.

    Args:
        sentence1 (str): The first sentence.
        sentence2 (str): The second sentence.

    Returns:
        float: The cosine similarity score between the two sentences. The score is between 0 and 1, 
               where 0 indicates no similarity and 1 indicates identical sentences.
    """
    # Convert sentences to lowercase and split into words
    words1 = sentence1.lower().split()
    words2 = sentence2.lower().split()

    # Build a vocabulary of unique words from both sentences
    unique_words = set(words1).union(set(words2))

    # Create frequency vectors for each sentence based on the vocabulary
    freq_vector1 = []
    freq_vector2 = []

    for word in unique_words:
        freq_vector1.append(words1.count(word))
        freq_vector2.append(words2.count(word))

    # Calculate the dot product of the two vectors
    dot_product = sum(f1 * f2 for f1, f2 in zip(freq_vector1, freq_vector2))

    # Calculate the magnitude of each vector
    magnitude1 = math.sqrt(sum(f ** 2 for f in freq_vector1))
    magnitude2 = math.sqrt(sum(f ** 2 for f in freq_vector2))

    # Handle the case when one of the magnitudes is zero (no overlap in words)
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Calculate and return cosine similarity
    return dot_product / (magnitude1 * magnitude2)

#Define sentence cosine similarity
def sim_sentence_cosine(df, query, feature):
    """Calculate the sentence cosine similarity between a query sentence and a sentence from the dataframe.

    Args:
        df (DataFrame): The dataframe containing the sentences.
        query (DataFrame): The query sentence.
        feature (str): The specific feature in the dataframe.

    Returns:
        DataFrame: A column containing the sentence cosine similarities.
    """
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate Euclidean distance between query value and each value in the feature column
    sent_cos_similarities = df[feature].apply(lambda x: sent_cosine_similarity(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(sent_cos_similarities, columns=[feature])

    return df[feature]

#Calculate raw vector cosine similarities
def vector_cosine_similarity(v1, v2):
    """
    Compute cosine similarity between two vectors. (For sentences use sent_cosine_similarity)

    Parameters:
        v1 (array-like): The first vector.
        v2 (array-like): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.

    Notes:
        Cosine similarity is a measure of similarity between two non-zero vectors of an inner product space.
        It is defined as the cosine of the angle between the two vectors.

        The cosine similarity ranges from -1 to 1, where 1 indicates that the vectors are identical,
        0 indicates that the vectors are orthogonal (i.e., have no similarity), and -1 indicates that the vectors
        are diametrically opposed (i.e., have maximum dissimilarity).

        This function assumes that the input vectors are non-zero and have the same length.
    """
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

#Define raw vector sim function
def sim_vector_cosine(df, query, feature):
    """
    Calculate the cosine similarity between a query vector and each vector in a feature column of a DataFrame.

    Args:
        df (DataFrame): The DataFrame containing the vector column.
        query (DataFrame): The DataFrame containing the query vector.
        feature (str): The name of the feature column.

    Returns:
        DataFrame: A DataFrame column with the cosine similarity values between the query vector and each vector in the feature column.
    """
    # Get the query value for the feature
    query_value = query[feature].iloc[0]

    # Calculate cosine similarity between query value and each value in the feature column
    vector_similarities = df[feature].apply(lambda x: vector_cosine_similarity(x, query_value))

    # Convert the Series to a DataFrame column with the feature name retained
    df[feature] = pd.DataFrame(vector_similarities, columns=[feature])

    return df[feature]
