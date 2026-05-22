import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/news.csv")

# Fill null values
df = df.fillna("")

# Create searchable text
df["text"] = (
    df["Headline"] + " " +
    df["News_body"] + " " +
    df["Source"]
)

print("Creating TF-IDF vectors...")

# Lightweight vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

vectors = vectorizer.fit_transform(df["text"])

# Save vectorizer
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save vectors
with open("tfidf_vectors.pkl", "wb") as f:
    pickle.dump(vectors, f)

# Save dataframe
with open("news.pkl", "wb") as f:
    pickle.dump(df, f)

print("TF-IDF model created successfully!")