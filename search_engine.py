import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load dataframe
with open("news.pkl", "rb") as f:
    df = pickle.load(f)

# Load TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load vectors
with open("tfidf_vectors.pkl", "rb") as f:
    vectors = pickle.load(f)

# Convert Date column
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)


def calculate_crop_score(text, crop_name):

    text = str(text).lower()
    crop = crop_name.lower()

    score = 0

    # Strong crop match
    score += text.count(crop) * 15

    agriculture_words = [
        "farming",
        "cultivation",
        "crop",
        "harvest",
        "yield",
        "farmer",
        "agriculture",
        "production",
        "market",
        "export",
        "disease"
    ]

    for word in agriculture_words:
        score += text.count(word)

    return score


def search_crop_news(crop_name, top_k=30):

    query = f"""
    {crop_name} farming agriculture
    {crop_name} crop production
    """

    # Transform query
    query_vector = vectorizer.transform([query])

    # Cosine similarity
    similarities = cosine_similarity(
        query_vector,
        vectors
    ).flatten()

    # Top indices
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []

    for idx in top_indices:

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

    # Sort by relevance
    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # Keep top 5 relevant
    top_results = results[:5]

    # Sort final selected results by date
    final_results = sorted(
        top_results,
        key=lambda x: x["date"]
        if pd.notnull(x["date"])
        else pd.Timestamp.min,
        reverse=True
    )

    return final_results