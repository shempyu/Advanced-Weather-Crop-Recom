from google import genai
import os
import json
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Create Gemini client
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def format_news_response(crop_name, news_results):

    formatted_news = ""

    # Prepare news
    for i, item in enumerate(news_results, start=1):

        formatted_news += f"""
News {i}:
Source: {item['source']}
Headline: {item['headline']}
Link: {item['link']}
"""

    # Prompt
    prompt = f"""
You are an agricultural news assistant.

User searched for crop: {crop_name}

Below are highly relevant crop-related articles.

Return ONLY valid JSON.

Format:

{{
  "headline1": "",
  "link1": "",

  "headline2": "",
  "link2": "",

  "headline3": "",
  "link3": "",

  "headline4": "",
  "link4": "",

  "headline5": "",
  "link5": ""
}}

Rules:
- Preserve original headlines
- Preserve original links
- Return latest relevant crop news first
- Return only JSON
- No markdown
- No explanations

Articles:
{formatted_news}
"""

    # Gemini response
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    # Clean output
    cleaned = response.text.replace(
        "```json", ""
    ).replace(
        "```", ""
    ).strip()

    # Convert to JSON
    parsed_json = json.loads(cleaned)

    return parsed_json