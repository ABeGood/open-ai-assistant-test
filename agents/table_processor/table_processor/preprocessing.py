import pandas as pd


def clean_dataframe(df):
  """
  Clean NaN values in a pandas dataframe

  Args:
      df: a pandas dataframe

  Returns:
      a new pandas dataframe with NaN values cleaned
  """
  # fill missing values with the mean of the column (numeric only)
  df = df.fillna(df.mean(numeric_only=True), inplace=False)

  # fill remaining missing values with the mode of the column
  for col in df.columns:
    df.loc[df[col].isnull(), col] = df[col].mode().iloc[0]

  return df


def drop_nan_dataframe(df):
  """
  Drop rows with NaN values in a pandas dataframe

  Args:
      df: a pandas dataframe

  Returns:
      a new pandas dataframe with NaN values dropped
  """
  return df.dropna()


def remove_spaces_in_column_names(df):
    """
    Remove spaces in column names of a pandas DataFrame.
    
    Parameters:
    df (pandas.DataFrame): Input DataFrame.
    
    Returns:
    pandas.DataFrame: DataFrame with spaces removed from column names.
    """
    # Create a dictionary to map old column names to new ones with spaces removed
    column_rename_dict = {col: col.strip().replace(' ', '') for col in df.columns}
    
    # Rename columns using the dictionary
    df = df.rename(columns=column_rename_dict)
    
    return df

def standardize_float_values(df):
    """
    Standardize all float values in a pandas DataFrame.

    Parameters:
    df (pandas.DataFrame): Input DataFrame.

    Returns:
    pandas.DataFrame: DataFrame with float values standardized.
    """
    # Iterate over columns
    for col in df.columns:
        # Check if the column contains float values
        if df[col].dtype == 'float64':
            # Standardize float values
            df[col] = (df[col] - df[col].mean()) / df[col].std()

    return df