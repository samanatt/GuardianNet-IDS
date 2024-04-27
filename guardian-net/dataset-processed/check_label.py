import pandas as pd


def count_records_with_label(df, label_value):
    """
    Count the number of records in a DataFrame where the 'label' column is equal to a specific value.

    Parameters:
        df (DataFrame): The input DataFrame.
        label_value: The specific value to check in the 'label' column.

    Returns:
        int: The number of records where the 'label' column is equal to the specified value.
    """
    return len(df[df['label'] == label_value])


def check_label_values(df):
    """
    Check if the 'label' column in the DataFrame contains only 0 and 1 values.

    Parameters:
        df (DataFrame): The input DataFrame.

    Returns:
        bool: True if the 'label' column contains only 0 and 1 values, False otherwise.
    """
    unique_labels = df['label'].unique()
    return set(unique_labels) == {0, 1}


# Example usage:
# Assuming df is your DataFrame
# Replace 'df' with the name of your DataFrame

# Example DataFrame
df = pd.read_csv(
    'C:\\Users\\Faraz\\PycharmProjects\\GuardianNet-IDS\\guardian-net\\dataset-processed\\CICIDS_train_binary.csv')

# Check if 'label' column contains only 0 and 1 values
label_values_valid = check_label_values(df)
if label_values_valid:
    print("The 'label' column contains only 0 and 1 values.")
    # Call the function to count records with label 0 and 1
    print(len(df))
    print("Number of records with label 1:", count_records_with_label(df, 1))
    print("Number of records with label 0:", count_records_with_label(df, 0))
else:
    print("The 'label' column contains other values beside 0 and 1!")
