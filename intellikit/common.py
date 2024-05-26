"""The common module contains common functions and classes used by the other modules.
"""
import pandas as pd
from multiprocessing import Pool, cpu_count


def retrieve_topk(cases, similarity_data, sim_column, k):
  """
  Ranks features in a DataFrame by total similarity and returns top k results.

  Args:
      cases (pandas.DataFrame): DataFrame containing features
      similarity_data(pandas.DataFrame): DataFrame containing similarity scores for each feature and a 'total_similarity' column.
      sim_column: the total similarity or weighted similarity column
      k (int): Number of top features to return.

  Returns:
      dataframe:  top k cases from the casebase.
  """

  #Combining the sets
  merged = pd.concat([cases, similarity_data], axis=1)

  # Sort DataFrame by 'total_similarity' in descending order
  sorting_df = merged.sort_values(by=sim_column, ascending=False)

  # Get the column to keep from the second DataFrame (assuming there's only one)
  column_to_keep = similarity_data.filter(like=sim_column).columns[0]  # Extract column name containing 'F'
  # Keep only the desired columns using list comprehension
  desired_columns = [col for col in merged.columns if col in cases.columns or col == column_to_keep]
  result_df = sorting_df[desired_columns]


  # Select top k features
  top_k_cases = result_df.head(k)

  return top_k_cases

def linearRetriever(df, query, similarity_functions, feature_weights, top_n=1):
    """
  linear retriever performs a K-NN search by computing all similarities between the query and each case one by one sequentially


  Args:
      df (pandas.DataFrame): DataFrame containing features for case characterization
      similarity_function (Dictionary): A dictionary containing similarity functions for each feature in the casebase.
      query (pandas.DataFrame): Datathe total similarity or weighted similarity column
      feature_weights (Dictionary): A dictionary of weights assigned to each feature
      top_n (int): Number of top similar cases to return.

  Returns:
      dataframe:  top k cases from the casebase.
  """
    # Create a DataFrame to store similarities
    similarities = pd.DataFrame(index=df.index)

    # Iterate over the columns in the DataFrame
    for feature in df.columns:
        # Check if the feature is in the similarity_functions dictionary
        if feature in similarity_functions:
            # Retrieve the similarity function for the feature
            similarity_function = similarity_functions[feature]

            # Check if feature weight is iterable, convert to float if necessary
            feature_weight = float(feature_weights.get(feature, 1.0))  # Default to 1.0 if not provided

            # Apply the similarity function to calculate similarities
            similarities[feature] = similarity_function(df[[feature]].copy(), query[[feature]], feature) * feature_weight
        else:
            # If the feature is not found in the similarity_functions dictionary, set the similarity to 0
            similarities[feature] = 0.0

    # Calculate total similarity as the sum of weighted similarities
    similarities['total_similarity'] = similarities.sum(axis=1)

    # Select top N cases with the lowest total similarity
    top_n_indices = similarities['total_similarity'].nlargest(top_n).index
    top_n_cases = df.loc[top_n_indices]

    return top_n_cases

#Parallel Linear Retrieval
# Define calculate_similarity function outside parallel_retriever
def calculate_similarity(feature, df, query, similarity_functions, feature_weights):
    return similarity_functions[feature](df[[feature]].copy(), query[[feature]], feature) * feature_weights[feature]

# Define parallel_retriever function
def parallelRetriever(df, query, similarity_functions, feature_weights, top_n=1):
    """
  Parallel linear retriever performs a K-NN search by computing all similarities between the query and each case using all available computing cors of the respective CPU.

  Args:
      df (pandas.DataFrame): DataFrame containing features for case characterization
      similarity_function (Dictionary): A dictionary containing similarity functions for each feature in the casebase.
      query (pandas.DataFrame): Datathe total similarity or weighted similarity column
      feature_weights (Dictionary): A dictionary of weights assigned to each feature
      top_n (int): Number of top similar cases to return.

  Returns:
      dataframe:  top k cases from the casebase.
  """
    with Pool(cpu_count()) as pool:
        # Use starmap to pass additional arguments to calculate_similarity
        similarity_results = pool.starmap(calculate_similarity, [(feature, df, query, similarity_functions, feature_weights) for feature in df.columns])

    similarities = pd.concat(similarity_results, axis=1)
    similarities['total_similarity'] = similarities.sum(axis=1)

    # Select top N cases with the lowest total similarity
    top_n_indices = similarities['total_similarity'].nlargest(top_n).index
    top_n_cases = df.loc[top_n_indices]

    return top_n_cases

# MACFAC Retrieval
def mac_stage(df, query, mac_features, similarity_functions, top_n_mac=2):
    mac_similarity = pd.DataFrame(index=df.index)

    for feature in mac_features:
        mac_similarity[feature] = similarity_functions[feature](df[[feature]].copy(), query[[feature]], feature)

    mac_similarity['total_similarity'] = mac_similarity.sum(axis=1)

    # Select top N candidates for the MAC stage
    top_n_indices = mac_similarity['total_similarity'].nlargest(top_n_mac).index
    return df.loc[top_n_indices]

def fac_stage(filtered_df, query, fac_features, similarity_functions, feature_weights, top_n_fac=1):
    similarities = pd.DataFrame(index=filtered_df.index)

    for feature in fac_features:
        similarities[feature] = similarity_functions[feature](filtered_df[[feature]].copy(), query[[feature]], feature) * feature_weights[feature]

    similarities['total_similarity'] = similarities.sum(axis=1)

    # Select top N cases for the FAC stage
    top_n_indices = similarities['total_similarity'].nlargest(top_n_fac).index
    top_n_cases = filtered_df.loc[top_n_indices]

    return top_n_cases

def macfacRetriever(df, query, mac_features, fac_features, similarity_functions, feature_weights, top_n_mac=2, top_n_fac=1):
    """
  MACFAC Retriever performs a two-staged retrieval where the first phase (MAC) uses a lightweight similarity to remove irrelevant cases for the second phase (FAC) where the final similarity for the filtered cases is evaluated.

  Args:
      df (pandas.DataFrame): DataFrame containing features for case characterization
      similarity_function (Dictionary): A dictionary containing similarity functions for each feature in the casebase.
      query (pandas.DataFrame): Datathe total similarity or weighted similarity column
      feature_weights (Dictionary): A dictionary of weights assigned to each feature
      mac_features: A list of features to use for the MAC phase (mac_features = ['feature4', 'feature5'])
      fac_features: A list of features to use for the FAC phase
      top_n_mac (int): Number of top similar cases to return during the MAC phase.
      top_n_fac (int): Number of top similar cases to return during the FAC phase

  Returns:
      dataframe:  top k cases from the casebase specified for the FAC phase.
  """
    filtered_df = mac_stage(df, query, mac_features, similarity_functions, top_n_mac)
    top_similar_cases = fac_stage(filtered_df, query, fac_features, similarity_functions, feature_weights, top_n_fac)
    return top_similar_cases
