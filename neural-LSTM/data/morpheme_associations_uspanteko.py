import pandas as pd
import ast
from process_uspanteko import uspanteko_data

# Load the CSV file into a DataFrame
df = uspanteko_data.copy()

print(len(df['segmented_text']) == len(df['gloss']))

# Create a new DataFrame where each row corresponds to a unique word-gloss pair
df = df.explode('segmented_text').reset_index(drop=True)
df = df.explode('gloss').reset_index(drop=True)

# print(df)

# Group this DataFrame by the gloss and aggregate the words associated with each gloss into a set
result = df.groupby('gloss')['segmented_text'].apply(set).reset_index()
result.to_csv('morpheme_word_assoc.csv', index=False)
