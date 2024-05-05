"""The common module contains common functions and classes used by the other modules.
"""
import pandas as pd


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
