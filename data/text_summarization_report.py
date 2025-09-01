import json
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os

nltk.download('punkt')

def summarize_section(sentences, top_n=3):
    if not sentences:
        return []

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    sentence_scores = tfidf_matrix.mean(axis=1).A1
    top_indices = sentence_scores.argsort()[::-1][:top_n]
    top_indices_sorted = sorted(top_indices)
    summary = [sentences[i] for i in top_indices_sorted]
    return summary

def summarize_transcript_sections(data, top_n=3):
    sections_to_summarize = ['guidance', 'forward_look', 'prior_year_mentions', 'year_over_year_mentions', 'compared_to_mentions', 'raising_mentions', 'refinance_mentions', 'production', 'outlook', 'demand_mentions', 'challenge_mentions']
    summary_results = {}

    for section in sections_to_summarize:
        text_list = data.get('metrics', {}).get(section, [])
        full_text = " ".join(text_list)
        sentences = sent_tokenize(full_text)
        summary = summarize_section(sentences, top_n=top_n)
        summary_results[section] = summary

    return summary_results

def format_metrics(metrics):
    # Convert string metrics to floats where possible
    keys_of_interest = ['revenue', 'eps', 'adjusted_ebitda', 'adjusted_ebitda_growth', 'free_cash_flow', 'net_income', 'operating_cash_flow', 'net_loss',
                        'monthly_active_users', 'daily_active_users', 'internet_customers', 'diluted_eps', 'adjusted_ebitda_margin', 'orders', 'operating_margin', 'segment_operating_margin', 'non_gaap_eps', 'consolidated_revenue', 'automotive_revenue', 'vehicles_produced', 'vehicles_delivered', 'gross_profit_loss', 'adjusted_ebitda_loss', 'equity_investment', 'vehicle_delivery_guidance', 'capex_guidance', 'net_revenue', 'adjusted_operating_income', 'industrial_free_cash_flow', 
        'tariff_impact', 'foreign_exchange_impact', 'total_inventories', 'gold_equivalent_production', 'all_in_sustaining_cost', 'cash_balance', 'adjusted_net_income', 'care_and_maintenance_cost', 'total_liquidity', 'marigold_output_ounces', 'ccv_production_ounces', 'seabee_output_ounces', 'seabee_aisc', 'puna_output_ounces', 'aisc', 'care_and_maintenance_costs', 'data_center_digital_alm_businesses', 'affo', 'affo_per_share', 'global_enterprise_revenue', 'net_income_per_share', 'cash_flow', 'cash_dividend', 'return_on_invested_capital', 'stock_repurchase', 'total_orders', 'marketplace_gov', 'net_revenue_margin', 'gaap_net_income','consolidated_net_sales', 
        'consolidated_operating_income', 'gaap_diluted_eps', 'adjusted_eps', 'orders', 'actual_eps', 'lumber_segment_adjusted_ebitda', 'available_liquidity', 'net_cash_balance', 'credit_facility', 'share_buybacks_and_dividends', 'combined_rate_ar6', 'total_net_sales', 'liquidity', 'interest_expense', 'total_available_liquidity']
    
    formatted = []
    for k in keys_of_interest:
        val = metrics.get(k, None)
        if val is None:
            continue  # Skip metrics that are N/A
        else:
            try:
                display_val = float(val)
                # Format large numbers nicely
                if display_val > 1000:
                    display_val = f"{display_val:,.0f}"
                else:
                    display_val = f"{display_val:.3f}"
            except Exception:
                display_val = str(val)
            formatted.append(f"{k.replace('_',' ').title()}: {display_val}")
    if not formatted:
        return "No financial metrics available."
    return "\n".join(formatted)

def format_sentiment(sentiment):
    # Format sentiment dict into readable string
    result = []
    for section, scores in sentiment.items():
        result.append(f"{section.title()}:")
        if scores is None:
            # If the sentiment section is None, show 0 for all categories
            for cat in ['neu', 'pos', 'neg']:
                result.append(f"  {cat.title()}: 0.000")
            continue
        for cat in ['neu', 'pos', 'neg']:
            val = scores.get(cat, 0)
            try:
                val_float = float(val)
            except (TypeError, ValueError):
                val_float = 0.0
            result.append(f"  {cat.title()}: {val_float:.3f}")
    return "\n".join(result)

def generate_report(transcript, top_n=3, output_folder='reports'):
    ticker = transcript.get('ticker', 'N/A')
    quarter = transcript.get('quarter', 'N/A')
    year = transcript.get('year', 'N/A')
    earnings_date = transcript.get('earnings_date', 'N/A')
    metrics = transcript.get('metrics', {})
    sentiment = transcript.get('finbert_sentiment', {})

    summaries = summarize_transcript_sections(transcript, top_n=top_n)

    report_lines = [
        f"Transcript Summary Report for {ticker} - {quarter} {year}",
        f"Earnings Date: {earnings_date}",
        "-"*60,
        "Key Financial Metrics:",
        format_metrics(metrics),
        "-"*60,
        "Sentiment Scores (FinBERT):",
        format_sentiment(sentiment),
        "-"*60,
        "Text Summaries:",
    ]

    for section, summary_sentences in summaries.items():
        report_lines.append(f"\n[{section.upper()}]")
        if summary_sentences:
            for s in summary_sentences:
                report_lines.append(f"- {s}")
        else:
            report_lines.append("- No content available")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    filename = f"{ticker}_{quarter}_{year}_summary_report.txt"
    filepath = os.path.join(output_folder, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    print(f"Report saved to: {filepath}")

def main():
    json_path = 'data/sentiment_enriched_transcripts.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        transcripts = json.load(f)

    # For this project, analyze one transcript at a time
    transcript = transcripts[0]

    generate_report(transcript, top_n=3)

if __name__ == "__main__":
    main()
