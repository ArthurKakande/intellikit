"""Main module."""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr
import numpy as np

def calculate_word_similarity(df, query, text_columns):
    """Calculate the similarity score between the query and the case. This basic function returns a value of 1 for all words that match and a value of zero when the words differ

    Args:
        df (dataframme, required): The dataframme containing the cases
        query (dictionary, required): The search query corresponding to a case you are looking for.
        text_columns (list, required): The columns from the dataframme for which to compare with the query values.

    Returns:
        dataframme: The dataframme of the calculated similarity values for each case against the query. 
    """    
    for column in text_columns:
        # Check for NaN values
        df[column] = df[column].fillna('')
        query[column] = query[column].fillna('')

        # Ensure the column is of string type
        if df[column].dtype != 'object':
            df[column] = df[column].astype(str)

        if query[column].dtype != 'object':
            query[column] = query[column].astype(str)

        # Check if documents are not empty or too short
        if df[column].str.len().sum() == 0 or query[column].str.len().sum() == 0:
            df[column] = 0  # Set similarity to 0 if documents are empty or too short
        else:
            # Compare words and set similarity accordingly
            df[column] = df[column].apply(lambda x: 1 if x == query[column].iloc[0] else 0)

    return df

def preprocess_data(df):
    label_encoder = LabelEncoder()

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna('')  # Fill missing values with an empty string
            df[column] = label_encoder.fit_transform(df[column])

    return df

def calculate_numerical_similarity(df, query, numerical_columns):
    for column in numerical_columns:
        df[column] = df.apply(lambda x: 1 / (1 + abs(x[column] - query[column].values[0])), axis=1)
    return df


def calculate_similarity(df, query, numerical_columns, text_columns):
    # Calculate similarity for numerical columns
    df = calculate_numerical_similarity(df, query, numerical_columns)

    # Calculate similarity for text columns
    df = calculate_word_similarity(df, query, text_columns)

    # Calculate similarity for color column
    #df = calculate_color_similarity(df, query, color_column)

    # Combine individual similarities into an overall similarity score
    df['similarity'] = df[numerical_columns + text_columns].mean(axis=1)

    return df

def retrieve_top_similar_results(df, similarity_column, top_k=6):
    top_similar_found = df.nlargest(top_k, similarity_column)
    return top_similar_found

def retrieve_final_results(df, casebase, similarity_column, top_k=6, order=False):
  similarity_file_path = 'similarity_output.csv'
  df_similarity = pd.read_csv(similarity_file_path)
  similarity_column = 'similarity'
  top_similar_found = retrieve_top_similar_results(df_similarity, similarity_column, top_k)
  top = top_similar_found
  final = casebase
  top['UniqueID'] = top.index + 0
  prefix = 'sim'
  toprenamed = top.add_prefix(prefix)
  final['UniqueID'] = final.index + 0
  final_extract = pd.merge(final,toprenamed,left_on='UniqueID',right_on='simUniqueID')
  final_sorted = final_extract.sort_values(by='simsimilarity', ascending=order)
  return final_sorted