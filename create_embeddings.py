import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/news.csv")

# Remove null values
df = df.fillna("")

# Create searchable text
df["text"] = (
    df["Headline"] + " " +
    df["News_body"] + " " +
    df["Source"]
)

print("Loading embedding model...")

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

# Generate embeddings
embeddings = model.encode(
    df["text"].tolist(),
    convert_to_numpy=True
)

# Convert to float32
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

# Save index
faiss.write_index(index, "news.index")

# Save dataframe
with open("news.pkl", "wb") as f:
    pickle.dump(df, f)

print("Embeddings created successfully!")