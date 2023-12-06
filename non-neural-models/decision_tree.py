import sys
sys.path.insert(0, '../')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Load the data
df = pd.read_csv('../data/uspanteko_data.csv')

print("Loaded df with shape: ", df.shape)
print(df.iloc[0:4])
# Flatten the lists in 'segmented_text' and 'gloss' columns
df = df.explode('segmented_text').reset_index(drop=True)
df = df.explode('gloss').reset_index(drop=True)

# Encode the 'segmented_text' and 'gloss' columns
le = LabelEncoder()
df['segmented_text_encoded'] = le.fit_transform(df['segmented_text'])
df['gloss_encoded'] = le.fit_transform(df['gloss'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['segmented_text_encoded'], df['gloss_encoded'], test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train.values.reshape(-1, 1), y_train)

# Evaluate the model
score = clf.score(X_test.values.reshape(-1, 1), y_test)
print(f"Model accuracy: {score * 100}%")