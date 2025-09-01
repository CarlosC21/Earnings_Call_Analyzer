import json
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon')

PARSED_FILE = "data/parsed_transcripts.json"
OUTPUT_FILE = "data/sentiment_enriched_transcripts.json"

def load_json(path):
    if not os.path.exists(path):
        print(f"⚠ {path} not found.")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_sentiment(text, analyzer):
    if not text:
        return None
    return analyzer.polarity_scores(text)

def analyze_section(sentences, analyzer):
    if not sentences:
        return None
    scores_list = [analyze_sentiment(s, analyzer) for s in sentences if s]
    if not scores_list:
        return None
    avg_scores = {}
    for key in scores_list[0].keys():
        avg_scores[key] = sum(score[key] for score in scores_list) / len(scores_list)
    return {k: round(v, 4) for k, v in avg_scores.items()}

def main():
    transcripts = load_json(PARSED_FILE)
    if transcripts is None:
        return

    sid = SentimentIntensityAnalyzer()
    enriched_transcripts = []

    for transcript in transcripts:
        metrics = transcript.get("metrics", {})

        # Per-section sentiment
        guidance_sentiment = analyze_section(metrics.get("guidance", []), sid)
        forward_look_sentiment = analyze_section(metrics.get("forward_look", []), sid)
        prior_year_sentiment = analyze_section(metrics.get("prior_year_mentions", []), sid)
        yoy_sentiment = analyze_section(metrics.get("year_over_year_mentions", []), sid)

        # Prepare full text by concatenating all relevant fields' sentences
        all_texts = []
        for key in ["guidance", "forward_look", "prior_year_mentions", "year_over_year_mentions"]:
            all_texts.extend(metrics.get(key, []))
        full_text = " ".join(all_texts)

        full_text_sentiment = analyze_sentiment(full_text, sid)

        transcript["finbert_sentiment"] = {
            "guidance": guidance_sentiment,
            "forward_look": forward_look_sentiment,
            "prior_year_mentions": prior_year_sentiment,
            "year_over_year_mentions": yoy_sentiment,
            "full_text": {k: round(v, 4) for k, v in full_text_sentiment.items()} if full_text_sentiment else None
        }

        enriched_transcripts.append(transcript)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(enriched_transcripts, f, indent=4, ensure_ascii=False)

    print(f"✅ Sentiment analysis complete with full-text. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
