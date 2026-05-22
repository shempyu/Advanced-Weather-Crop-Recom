from flask import Flask, request, jsonify
from flask_cors import CORS
from search_engine import search_crop_news
from llm_formatter import format_news_response
import os

app = Flask(__name__)
CORS(app)

@app.route("/crop-news", methods=["POST"])
def crop_news():

    try:

        data = request.get_json()

        crop = data.get("crop")

        if not crop:
            return jsonify({
                "error": "Crop name is required"
            }), 400

        # Search news
        news_results = search_crop_news(crop)

        # Format using Gemini
        formatted_response = format_news_response(
            crop,
            news_results
        )

        return jsonify({
            "crop": crop,
            "results": formatted_response
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Crop News API Running"
    })


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 10000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )