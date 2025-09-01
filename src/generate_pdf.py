# generate_pdf.py
import json
import ast
import re
from datetime import datetime
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors


# ---------------- utilities ---------------- #
def format_key(key: str) -> str:
    return " ".join(word.capitalize() for word in str(key).split("_"))


def normalize_escapes(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\\r", " ").replace("\\n", " ").replace("\\t", " ")
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(s.split()).strip()


def strip_surrounding_quotes_brackets(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
        s = s[1:-1].strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()
    return s.strip(" \t\n'\"[]()")


def manual_top_level_split_list(s: str):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s_inner = s[1:-1]
    else:
        s_inner = s

    items, cur = [], []
    in_quote, quote_char, escape = False, None, False
    for ch in s_inner:
        if escape:
            cur.append(ch)
            escape = False
            continue
        if ch == "\\":
            cur.append(ch)
            continue
        if in_quote:
            cur.append(ch)
            if ch == quote_char:
                in_quote = False
                quote_char = None
            continue
        else:
            if ch in ("'", '"'):
                in_quote, quote_char = True, ch
                cur.append(ch)
                continue
            if ch == ",":
                items.append("".join(cur).strip())
                cur = []
                continue
            else:
                cur.append(ch)
    last = "".join(cur).strip()
    if last:
        items.append(last)
    return [it.strip() for it in items if it.strip()]


def try_parse_list_like(raw_str: str):
    if not isinstance(raw_str, str):
        return None
    candidate = raw_str.strip()
    if not (candidate.startswith("[") and candidate.endswith("]")):
        return None

    candidate_norm = normalize_escapes(candidate)
    try:
        parsed = json.loads(candidate_norm)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass
    items = manual_top_level_split_list(candidate)
    if items:
        return items
    return None


def split_into_sentences(text: str):
    if not text:
        return []
    text = " ".join(text.split())
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def normalize_field_to_sentences(raw):
    if raw is None:
        return []
    if isinstance(raw, list):
        raw_items = [str(x) for x in raw]
    elif isinstance(raw, str):
        parsed_list = try_parse_list_like(raw)
        raw_items = parsed_list if parsed_list is not None else [raw]
    else:
        raw_items = [str(raw)]

    sentences_out = []
    for item in raw_items:
        item = normalize_escapes(item)
        item = strip_surrounding_quotes_brackets(item)
        item = " ".join(item.split())
        if not item:
            continue
        pieces = split_into_sentences(item)
        for p in pieces:
            p = strip_surrounding_quotes_brackets(p)
            p = " ".join(p.split())
            if p:
                sentences_out.append(p)
    return sentences_out


# ---------------- Metric Keys ---------------- #
METRIC_KEYS = {
    'revenue', 'eps', 'adjusted_ebitda', 'adjusted_ebitda_growth', 'free_cash_flow',
    'net_income', 'operating_cash_flow', 'net_loss', 'monthly_active_users',
    'daily_active_users', 'internet_customers', 'diluted_eps', 'adjusted_ebitda_margin',
    'orders', 'operating_margin', 'segment_operating_margin', 'non_gaap_eps',
    'consolidated_revenue', 'automotive_revenue', 'vehicles_produced', 'vehicles_delivered',
    'gross_profit_loss', 'adjusted_ebitda_loss', 'equity_investment',
    'vehicle_delivery_guidance', 'capex_guidance', 'net_revenue',
    'adjusted_operating_income', 'industrial_free_cash_flow', 'tariff_impact',
    'foreign_exchange_impact', 'total_inventories', 'gold_equivalent_production',
    'all_in_sustaining_cost', 'cash_balance', 'adjusted_net_income',
    'care_and_maintenance_cost', 'total_liquidity', 'marigold_output_ounces',
    'ccv_production_ounces', 'seabee_output_ounces', 'seabee_aisc',
    'puna_output_ounces', 'aisc', 'care_and_maintenance_costs',
    'data_center_digital_alm_businesses', 'affo', 'affo_per_share',
    'global_enterprise_revenue', 'net_income_per_share', 'cash_flow', 'cash_dividend',
    'return_on_invested_capital', 'stock_repurchase', 'total_orders', 'marketplace_gov',
    'net_revenue_margin', 'gaap_net_income', 'consolidated_net_sales',
    'consolidated_operating_income', 'gaap_diluted_eps', 'adjusted_eps',
    'actual_eps', 'lumber_segment_adjusted_ebitda', 'available_liquidity',
    'net_cash_balance', 'credit_facility', 'share_buybacks_and_dividends',
    'combined_rate_ar6', 'total_net_sales', 'liquidity', 'interest_expense',
    'total_available_liquidity'
}


# ---------------- PDF generation ---------------- #
def create_pdf(enriched_json, price_data_json, output_path="final_report.pdf"):
    with open(enriched_json, "r", encoding="utf-8") as f:
        transcripts = json.load(f)
    with open(price_data_json, "r", encoding="utf-8") as f:
        price_data = json.load(f)

    doc = SimpleDocTemplate(output_path, pagesize=LETTER)
    styles = getSampleStyleSheet()

    centered_big = styles["Normal"].clone("CenteredBig")
    centered_big.alignment = TA_CENTER
    centered_big.fontSize = 12
    centered_big.leading = 16

    centered_header = styles["Heading3"].clone("CenteredHeader")
    centered_header.alignment = TA_CENTER
    centered_normal = styles["Normal"].clone("CenteredNormal")
    centered_normal.alignment = TA_CENTER

    # Smaller font for numeric metrics
    metric_style = styles["Normal"].clone("MetricStyle")
    metric_style.alignment = TA_CENTER
    metric_style.fontSize = 10
    metric_style.leading = 12

    left_header = styles["Heading4"].clone("LeftHeader")
    left_header.alignment = TA_LEFT
    left_normal = styles["Normal"].clone("LeftNormal")
    left_normal.alignment = TA_LEFT

    elements = []

    for idx, data in enumerate(transcripts):
        ticker = data.get("ticker", "N/A")
        earnings_date_str = data.get("earnings_date", "N/A")
        try:
            earnings_date_fmt = datetime.strptime(
                earnings_date_str, "%Y-%m-%d"
            ).strftime("%B %d %Y")
        except Exception:
            earnings_date_fmt = earnings_date_str

        elements.append(Paragraph("<b>Earnings Transcript Summary</b>", styles["Title"]))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(f"Ticker: {ticker}", centered_big))
        elements.append(Paragraph(f"Quarter: {data.get('quarter','N/A')}", centered_big))
        elements.append(Paragraph(f"Year: {data.get('year','N/A')}", centered_big))
        elements.append(Paragraph(f"Earnings Date: {earnings_date_fmt}", centered_big))
        elements.append(Spacer(1, 16))

        # Stock analysis
        if ticker in price_data:
            summary = price_data[ticker].get("summary", {})
            elements.append(Paragraph("<b>Stock Analysis</b>", centered_header))
            elements.append(Spacer(1, 6))

            def format_percent(value):
                try:
                    v = float(value)
                except Exception:
                    return "N/A"
                color = "green" if v >= 0 else "red"
                return f'<font color="{color}">{v:+.2f}%</font>'

            gain = summary.get("largest_gain", {})
            elements.append(
                Paragraph(
                    f"<b>Largest Gain:</b> {gain.get('date','N/A')} ({format_percent(gain.get('percent'))})",
                    centered_normal,
                )
            )
            drop = summary.get("largest_drop", {})
            elements.append(
                Paragraph(
                    f"<b>Largest Drop:</b> {drop.get('date','N/A')} ({format_percent(drop.get('percent'))})",
                    centered_normal,
                )
            )
            elements.append(
                Paragraph(
                    f"<b>Earnings Day ({summary.get('earnings_date','N/A')}):</b> {format_percent(summary.get('earnings_day_change'))} ",
                    centered_normal,
                )
            )
            elements.append(Spacer(1, 20))

        # ✅ Key Metrics - center numeric metrics
        metrics = data.get("metrics", {})
        if isinstance(metrics, dict):
            clean_metrics = {
                k: v
                for k, v in metrics.items()
                if v not in [None, "", "N/A", []] and not (isinstance(v, list) and len(v) == 0)
            }
            if clean_metrics:
                elements.append(Paragraph("<b>Key Metrics</b>", centered_header))
                elements.append(Spacer(1, 6))
                for k, v in clean_metrics.items():
                    sentences = normalize_field_to_sentences(v)
                    if not sentences:
                        continue

                    if k.lower() in METRIC_KEYS:
                        elements.append(Paragraph(f"<b>{format_key(k)}:</b>", centered_header))
                        elements.append(Spacer(1, 4))
                        for s in sentences:
                            elements.append(Paragraph(s.strip(), metric_style))
                            elements.append(Spacer(1, 3))
                    else:
                        elements.append(Paragraph(f"<b>{format_key(k)}:</b>", left_header))
                        elements.append(Spacer(1, 4))
                        for s in sentences:
                            elements.append(Paragraph(f"• {s.strip()}", left_normal))
                            elements.append(Spacer(1, 3))

                    elements.append(Spacer(1, 8))
                elements.append(Spacer(1, 20))

        # Narrative sections
        narrative_keys = [
            "guidance", "forward_look", "prior_year_mentions", "year_over_year_mentions",
            "compared_to_mentions", "raising_mentions", "refinance_mentions", "production",
            "outlook", "demand_mentions", "challenge_mentions"
        ]
        for key in narrative_keys:
            raw = data.get(key, None)
            sentences = normalize_field_to_sentences(raw)
            if not sentences:
                continue
            elements.append(Paragraph(f"<b>{format_key(key)}:</b>", left_header))
            elements.append(Spacer(1, 6))
            for s in sentences:
                if s.strip():
                    elements.append(Paragraph(f"• {s.strip()}", left_normal))
                    elements.append(Spacer(1, 4))
            elements.append(Spacer(1, 10))

        # Sentiment (as table)
        sentiment = data.get("finbert_sentiment", {})
        if sentiment:
            elements.append(Paragraph("<b>Sentiment Analysis</b>", centered_header))
            elements.append(Spacer(1, 12))

            table_data = [["Section", "Compound", "Positive", "Neutral", "Negative"]]
            for sec, scores in sentiment.items():
                if not scores:
                    continue
                table_data.append([
                    format_key(sec),
                    scores.get("compound", "N/A"),
                    scores.get("pos", "N/A"),
                    scores.get("neu", "N/A"),
                    scores.get("neg", "N/A"),
                ])

            sentiment_table = Table(table_data, hAlign="CENTER")
            sentiment_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]))

            elements.append(sentiment_table)
            elements.append(Spacer(1, 20))

        if idx < len(transcripts) - 1:
            elements.append(PageBreak())

    doc.build(elements)
    print(f"✅ PDF created at {output_path}")


if __name__ == "__main__":
    create_pdf("data/sentiment_enriched_transcripts.json", "data/price_data.json", "final_report.pdf")
