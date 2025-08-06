# Script 

import os
import re
import json

# Directory for manually downloaded transcripts
TRANSCRIPT_DIR = "./data/transcripts/"
OUTPUT_FILE = "./data/parsed_transcripts.json"

def clean_text(text: str) -> str:
    """
    Cleans raw transcript text:
    - Removes timestamps
    - Removes speaker labels (Operator:, CEO:)
    - Removes non-verbal actions [laughter]
    """
    # Remove timestamps like "00:12"
    text = re.sub(r"\b\d{1,2}:\d{2}\b", "", text)

    # Remove bracketed actions like [laughter]
    text = re.sub(r"\[.*?\]", "", text)

    # Remove speaker labels (Operator:, CFO:, CEO:)
    text = re.sub(r"^\s*[A-Za-z\s]+:\s*", "", text, flags=re.MULTILINE)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def parse_transcripts():
    parsed_data = {}

    for file in os.listdir(TRANSCRIPT_DIR):
        if file.endswith(".txt"):
            ticker = file.split("_")[0]      # Extract ticker (e.g., AAPL)
            quarter = file.replace(".txt", "")  # Keep filename without extension

            with open(os.path.join(TRANSCRIPT_DIR, file), "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned_text = clean_text(raw_text)

            # Optional: Split into sentences
            sentences = re.split(r'(?<=[.!?]) +', cleaned_text)

            parsed_data[quarter] = {
                "ticker": ticker,
                "quarter": quarter,
                "raw_text": raw_text,
                "cleaned_text": cleaned_text,
                "sentences": sentences
            }

    # Save to JSON for later NLP processing
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(parsed_data, out, indent=4)

    print(f"✅ Parsed transcripts saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    parse_transcripts()
