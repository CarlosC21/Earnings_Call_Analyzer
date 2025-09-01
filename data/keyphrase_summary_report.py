import json
import yake
import nltk
from nltk.tokenize import sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

nltk.download('punkt')
nltk.download('vader_lexicon')

def extract_key_phrases(text, max_phrases=15):
    kw_extractor = yake.KeywordExtractor(lan="en", n=3, top=max_phrases)
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, score in sorted(keywords, key=lambda x: x[1])]

def find_keyword_contexts(text, keywords, window=2):
    sentences = sent_tokenize(text)
    sentences_lower = [s.lower() for s in sentences]
    keyword_contexts = {}

    for kw in keywords:
        kw_lower = kw.lower()
        indices = [i for i, sent in enumerate(sentences_lower) if kw_lower in sent]
        contexts = []
        for idx in indices:
            start = max(0, idx - window)
            end = min(len(sentences), idx + window + 1)
            snippet = " ".join(sentences[start:end])
            contexts.append(snippet)
        if contexts:
            unique_contexts = []
            seen = set()
            for c in contexts:
                if c not in seen:
                    unique_contexts.append(c)
                    seen.add(c)
                if len(unique_contexts) >= 3:
                    break
            keyword_contexts[kw] = unique_contexts
    return keyword_contexts

def analyze_sentiment_for_phrases(keyword_contexts):
    sia = SentimentIntensityAnalyzer()
    phrase_sentiments = {}

    for phrase, snippets in keyword_contexts.items():
        pos_scores = []
        neu_scores = []
        neg_scores = []
        compound_scores = []
        for snippet in snippets:
            sents = sent_tokenize(snippet)
            for sent in sents:
                if phrase.lower() in sent.lower():
                    scores = sia.polarity_scores(sent)
                    pos_scores.append(scores['pos'])
                    neu_scores.append(scores['neu'])
                    neg_scores.append(scores['neg'])
                    compound_scores.append(scores['compound'])
        if pos_scores:
            phrase_sentiments[phrase] = {
                'positive_avg': sum(pos_scores) / len(pos_scores),
                'neutral_avg': sum(neu_scores) / len(neu_scores),
                'negative_avg': sum(neg_scores) / len(neg_scores),
                'compound_avg': sum(compound_scores) / len(compound_scores),
                'mention_count': len(pos_scores)
            }
        else:
            phrase_sentiments[phrase] = {
                'positive_avg': 0.0,
                'neutral_avg': 0.0,
                'negative_avg': 0.0,
                'compound_avg': 0.0,
                'mention_count': 0
            }
    return phrase_sentiments

def generate_keyphrase_summary_report(text, max_phrases=15):
    key_phrases = extract_key_phrases(text, max_phrases=max_phrases)
    contexts = find_keyword_contexts(text, key_phrases)
    phrase_sentiments = analyze_sentiment_for_phrases(contexts)
    return key_phrases, contexts, phrase_sentiments

def save_report_pdf(ticker, quarter, year, key_phrases, contexts, phrase_sentiments, output_dir="reports"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{ticker}_{year}_{quarter}_keyphrase_summary.pdf"
    path = os.path.join(output_dir, filename)

    doc = SimpleDocTemplate(path, pagesize=letter,
                            rightMargin=72,leftMargin=72,
                            topMargin=72,bottomMargin=72)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=4))  # Justify alignment

    story = []

    title = f"Key Phrase Summary Report for {ticker} {quarter} {year}"
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 12))

    # Key Phrases list
    story.append(Paragraph("Top Key Phrases:", styles['Heading2']))
    for i, phrase in enumerate(key_phrases, 1):
        story.append(Paragraph(f"{i}. {phrase}", styles['Normal']))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Keyword-in-Context Highlights with Sentiment Breakdown:", styles['Heading2']))
    story.append(Spacer(1, 6))

    for phrase in key_phrases:
        sentiment = phrase_sentiments.get(phrase, {})
        story.append(Paragraph(f"<b>{phrase}</b>", styles['Heading3']))
        story.append(Paragraph(f"Mentions: {sentiment.get('mention_count', 0)}", styles['Normal']))
        story.append(Paragraph(f"Avg Positive Sentiment: {sentiment.get('positive_avg', 0):.3f}", styles['Normal']))
        story.append(Paragraph(f"Avg Neutral Sentiment: {sentiment.get('neutral_avg', 0):.3f}", styles['Normal']))
        story.append(Paragraph(f"Avg Negative Sentiment: {sentiment.get('negative_avg', 0):.3f}", styles['Normal']))
        story.append(Paragraph(f"Avg Compound Score: {sentiment.get('compound_avg', 0):.3f}", styles['Normal']))
        story.append(Spacer(1, 4))

        snippets = contexts.get(phrase, [])
        if snippets:
            for idx, snippet in enumerate(snippets, 1):
                snippet_paragraph = Paragraph(f"[{idx}] ...{snippet}...", styles['Justify'])
                story.append(snippet_paragraph)
                story.append(Spacer(1, 6))
        else:
            story.append(Paragraph("No context snippets found.", styles['Normal']))
        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"PDF report saved: {path}")

def load_transcripts(json_path):
    print(f"Loading data from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    return data

def main():
    json_path = "data/sentiment_enriched_transcripts.json"  # adjust path as needed
    transcripts = load_transcripts(json_path)

    for record in transcripts:
        ticker = record.get("ticker", "UNKNOWN")
        quarter = record.get("quarter", "UNKNOWN")
        year = record.get("year", "UNKNOWN")
        metrics = record.get("metrics", {})

        combined_text_parts = []
        for field in ['guidance', 'forward_look', 'prior_year_mentions', 'year_over_year_mentions']:
            items = metrics.get(field, [])
            if isinstance(items, list):
                combined_text_parts.extend(items)
            elif isinstance(items, str):
                combined_text_parts.append(items)
        combined_text = " ".join(combined_text_parts).strip()

        if not combined_text:
            print(f"No narrative text found for {ticker} {quarter} {year}, skipping...")
            continue

        print(f"\nProcessing {ticker} {quarter} {year}...")
        key_phrases, contexts, phrase_sentiments = generate_keyphrase_summary_report(combined_text, max_phrases=15)
        save_report_pdf(ticker, quarter, year, key_phrases, contexts, phrase_sentiments)

if __name__ == "__main__":
    main()
