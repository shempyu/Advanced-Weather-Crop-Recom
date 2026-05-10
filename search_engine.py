import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index("news.index")

# Load dataframe
with open("news.pkl", "rb") as f:
    df = pickle.load(f)

# Convert Date column
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_crop_score(text, crop_name):

    text = str(text).lower()
    crop = crop_name.lower()

    score = 0

    # Strong crop match
    score += text.count(crop) * 15

    # Agriculture-related keywords
    agriculture_words = [
        "farming",
        "cultivation",
        "crop",
        "harvest",
        "yield",
        "farmer",
        "agriculture",
        "seed",
        "soil",
        "irrigation",
        "production",
        "disease",
        "fertilizer",
        "pesticide",
        "msp",
        "market",
        "export"
    ]

    for word in agriculture_words:
        score += text.count(word)

    return score


def search_crop_news(crop_name, top_k=30):

    # Better semantic query
    query = f"""
    {crop_name} farming cultivation agriculture
    {crop_name} crop production harvest
    {crop_name} farmer market export news
    """

    # Generate embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    # FAISS search
    distances, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    # Process search results
    for idx in indices[0]:

        row = df.iloc[idx]

        combined_text = f"""
        {row['Headline']}
        {row['News_body']}
        """

        crop_score = calculate_crop_score(
            combined_text,
            crop_name
        )

        results.append({
            "source": row["Source"],
            "headline": row["Headline"],
            "link": row["Link"],
            "body": row["News_body"],
            "date": row["Date"],
            "score": crop_score
        })

    # STEP 1:
    # Sort by relevance score
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # STEP 2:
    # Keep top 5 relevant results
    top_results = results[:5]

    # STEP 3:
    # Sort ONLY final selected results by latest date
    final_results = sorted(
        top_results,
        key=lambda x: x["date"]
        if pd.notnull(x["date"])
        else pd.Timestamp.min,
        reverse=True
    )

    return final_results