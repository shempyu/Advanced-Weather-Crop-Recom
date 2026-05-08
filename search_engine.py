import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("newss.index")

# Load dataframe
with open("newss.pkl", "rb") as f:
    df = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def search_crop_news(crop_name, top_k=5):

    # Create query
    query = f"{crop_name} farming agriculture crop news"

    # Generate embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.array(query_embedding).astype("float32")

    # Search FAISS
    distances, indices = index.search(query_embedding, top_k)

    results = []

    for idx in indices[0]:

        row = df.iloc[idx]

        results.append({
            "source": row["Source"],
            "headline": row["Headline"],
            "link": row["Link"],
            "body": row["News_body"]
        })

    return results