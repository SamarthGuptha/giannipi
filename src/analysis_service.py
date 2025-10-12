from typing import Dict, List, Any

QUOTE_CATEGORIES = ['championship_quotes', "funny_quotes"]

def analyze_quotes(data:Dict[str, Any]) -> Dict[str, Any]:
    analysis_results = {
        "total_quotes": 0,
        "distribution_by_source": {},
        "average_length_by_source": {}
    }

    all_quotes_count = 0

    for category in QUOTE_CATEGORIES:
        quotes = data.get(category, [])
        category_count = len(quotes)

        analysis_results["distribution_by_source"][category] = category_count
        all_quotes_count += category_count

        total_word_count = 0

        for quote_obj in quotes:
            quote_text = quote_obj.get('quote', '')
            word_count = len(quote_text.split())
            total_word_count += word_count

        if category_count>0:
            avg_length = round(total_word_count/category_count, 2)
        else: avg_length = 0

        analysis_results["average_length_by_source"][category] = avg_length
    analysis_results["total_quotes"] = all_quotes_count

    return analysis_results

if __name__ == '__main__':
    print("Quote Analysis service loaded.")