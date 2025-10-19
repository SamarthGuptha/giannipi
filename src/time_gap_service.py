import json, math
from datetime import datetime
from collections import defaultdict

STAT_CATEGORIES = ['points', 'rebounds', 'assists', 'steals', 'blocks']

def analyze_rest_period(days_rest):
    if days_rest ==1: return "Back-to-Back (1 Day Rest)"
    elif days_rest == 2: return "Normal Rest (2 Days Rest)"
    elif 3<= days_rest <=6: return "Extended Rest (3-6 Days rest)"
    else: return "Long Break (7+ Days rest)"

def analyze_time_gaps(stat_lines):
    if not stat_lines: return None, "Stat lines data is empty."

    try:
        sorted_games = sorted(
            stat_lines,
            key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d')
        )
    except (ValueError, KeyError) as e:
        return None, f"Error parsing date format or missing 'date' key: {e}"

    rest_period_data = defaultdict(lambda: {'games': 0, 'totals': defaultdict(int)})

    for i in range(1, len(sorted_games)):
        current_game = sorted_games[i]
        previous_game = sorted_games[i-1]

        current_date = datetime.strptime(current_game['date'], '%Y-%m-%d')
        previous_date = datetime.strptime(previous_game['date'], '%Y-%m-%d')

        days_difference = (current_date - previous_date).days
        days_rest = days_difference - 1

        if days_rest<1: continue

        rest_category = analyze_rest_period(days_rest)

        stats = current_game.get('stats', {})
        category_data = rest_period_data[rest_category]

        category_data['games'] +=1

        for stat in STAT_CATEGORIES:
            category_data['totals'][stat]+=stats.get(stat, 0)

    results = {}
    for category, data in rest_period_data.items():
        if data['games']>0:
            averages = {
                stat: round(data['totals'][stat] / data['games'], 2)
                for stat in STAT_CATEGORIES
            }
            results[category] = {
                "games_played": data['games'],
                "average_performance": averages

            }
    if not results: return None, "Not enough chronological data to calculate rest periods (less than 2 games)."

    return results, None

if __name__ == '__main__':
    DATA_FILE = 'giannis_data.json'
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
            stat_lines = data.get('stat_lines', [])

        analysis, error = analyze_time_gaps(stat_lines)

        if error:
            print(f"Error during local analysis: {error}")
        else:
            print("TIME GAP ANALYSIS RESULTS!")
            print(json.dumps(analysis, indent=4))

    except FileNotFoundError:
        print(f"CRITICAL ERROR: {DATA_FILE} not found. Cannot run local test.")
    except Exception as e:
        print(f"An Unexpected error occurred during execution: {e}")
