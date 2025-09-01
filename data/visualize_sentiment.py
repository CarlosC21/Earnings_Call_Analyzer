import json
import matplotlib.pyplot as plt
import numpy as np

def visualize_sentiment(data):
    entry = data[0]
    sentiments = entry['finbert_sentiment']

    categories = ['guidance', 'forward_look', 'prior_year_mentions', 'year_over_year_mentions', 'full_text']

    # Safely handle None values
    positive = [sentiments[cat]['pos'] if sentiments.get(cat) else 0 for cat in categories]
    neutral = [sentiments[cat]['neu'] if sentiments.get(cat) else 0 for cat in categories]
    negative = [sentiments[cat]['neg'] if sentiments.get(cat) else 0 for cat in categories]

    n_groups = len(categories)
    group_width = 0.6  # total width for all bars in a group
    bar_width = group_width / 3  # width of each bar

    # Create positions for groups spaced wider apart
    x = np.arange(n_groups) * (group_width + 0.4)  # add 0.4 for space between groups

    fig, ax = plt.subplots(figsize=(10, 5))

    bars1 = ax.bar(x - bar_width, positive, bar_width, label='Positive', color='green')
    bars2 = ax.bar(x, neutral, bar_width, label='Neutral', color='gray')
    bars3 = ax.bar(x + bar_width, negative, bar_width, label='Negative', color='red')

    ax.set_ylabel('Sentiment Score')
    ax.set_title(f"FinBERT Sentiment Breakdown - {entry['ticker']} {entry['quarter']} {entry['year']}")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha='right')
    ax.legend()

    # Annotate bars with values
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0,3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

def main():
    with open('data/sentiment_enriched_transcripts.json', 'r') as f:
        data = json.load(f)
    visualize_sentiment(data)

if __name__ == "__main__":
    main()
