from typing import Dict, Any

QUOTE_CATEGORIES = ['championship_quotes', 'funny_quotes']

def analyze_speakers(data: Dict[str, Any]) -> Dict[str, Any]:
    speaker_counts: Dict[str, int] = {}

    for category in QUOTE_CATEGORIES:
        quotes = data.get(category, [])

        for quote_obj in quotes:
            speaker = quote_obj.get('speaker', 'Unknown Speaker')
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1


        analysis_results = {
            "unique_speakers": sorted(list(speaker_counts.keys())),
            "total_quotes_analyzed": sum(speaker_counts.values()),
            "quote_counts_by_speaker": speaker_counts
        }

        return analysis_results

if __name__ == '__main__':
    print("Speaker Analysis service loaded.")