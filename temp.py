# from datasets import load_from_disk
#
# # Load dataset from disk
# dataset_path = 'dataset/dataset_test'
# ds = load_from_disk(dataset_path)
#
# # Print dataset structure to inspect
# print(ds)
#
# # Save the entire dataset (no splits) to a CSV file
# ds.to_csv("dataset/dataset_test_csv/dataset_test.csv")

import pandas as pd

# read_csv function which is used to read the required CSV file
data = pd.read_csv('dataset/dataset_test_csv/dataset_test.csv')

# display
print("Original 'input.csv' CSV Data: \n")


# drop function which is used in removing or deleting rows or columns from the CSV files
data.drop('target', inplace=True, axis=1)

for i, j in data[:10].iterrows():
    print(j['text'], i)

