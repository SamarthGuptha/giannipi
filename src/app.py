import json
import random
from flask import Flask, jsonify, request
from datetime import datetime

app = Flask(__name__)

DATA_FILE = 'giannis_data.json'
try:
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    DATA_LOADED = True
except FileNotFoundError:
    print(f"CRITICAL ERROR: {DATA_FILE} not found. Please ensure it is in the same directory as app.py.")
    data = {
        "stat_lines": [],
        "dunks_by_type": [],
        "championship_quotes": [],
        "funny_quotes": []
    }
    DATA_LOADED = False
STAT_CATEGORIES = ['points', 'rebounds', 'assists', 'steals', 'blocks']

def check_data_ready(key=None):
    if not DATA_LOADED:
        return jsonify({"error": f"Critical Error: Data file '{DATA_FILE}' could not be loaded. Ensure the file exists next to app.py."}), 500

    if key and not data.get(key):
        return jsonify({"error": f"Data source '{key}' is empty or missing in {DATA_FILE}."}), 404

    return None


def filter_stat_lines(stat_lines, args):
    filtered_lines = []
    for line in stat_lines:
        keep = True
        for stat in STAT_CATEGORIES:
            min_val = args.get(f'{stat}_min', type=int)
            if min_val is not None and line['stats'].get(stat, 0) < min_val:
                keep = False
                break
        if keep:
            filtered_lines.append(line)

    date_start_str = args.get('date_start')
    date_end_str = args.get('date_end')

    if date_start_str or date_end_str:
        final_lines = []
        try:
            start_date = datetime.strptime(date_start_str, '%Y-%m-%d') if date_start_str else datetime.min
            end_date = datetime.strptime(date_end_str, '%Y-%m-%d') if date_end_str else datetime.max

            for line in filtered_lines:
                game_date = datetime.strptime(line['date'], '%Y-%m-%d')
                if start_date <= game_date <= end_date:
                    final_lines.append(line)

            return final_lines

        except ValueError:
            return [{"error": "Invalid date format. Use YYYY-MM-DD."}]

    return filtered_lines


def sort_stat_lines(stat_lines, args):

    sort_by = args.get('sort_by')
    order = args.get('order', 'desc').lower()

    if sort_by and sort_by in STAT_CATEGORIES + ['date']:

        if sort_by == 'date':
            sort_key = lambda x: datetime.strptime(x['date'], '%Y-%m-%d')
        else:
            sort_key = lambda x: x['stats'].get(sort_by, 0)

        try:
            return sorted(
                stat_lines,
                key=sort_key,
                reverse=(order == 'desc')
            )
        except Exception as e:
            return [{"error": f"Sorting failed: {str(e)}"}]

    return stat_lines


def is_double_double(stats):
    double_count = 0
    for stat in STAT_CATEGORIES:
        if stats.get(stat, 0) >= 10 and stat != 'blocks' and stat != 'steals':
            double_count += 1

    return double_count >= 2


def is_triple_double(stats):
    triple_count = 0
    for stat in STAT_CATEGORIES:
        if stats.get(stat, 0) >= 10:
            triple_count += 1

    return triple_count >= 3
@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the giannipi!",
        "endpoints": [
            "/giannis/stat-lines",
            "/giannis/career-averages",
            "/giannis/stats-by-opponent?opponent=...",
            "/giannis/doubles?type=dd",
            "/search/quotes?query=...&source=...",
            "/giannis/dunks-by-type",
            "/giannis/dunks/count",
            "/bucks/championship-quotes",
            "/giannis/funny-quotes",
        ]
    })


@app.route('/giannis/stat-lines')
def get_stat_lines():
    error_response = check_data_ready('stat_lines')
    if error_response:
        return error_response

    if not request.args:
        return jsonify(random.choice(data['stat_lines']))

    filtered = filter_stat_lines(data['stat_lines'], request.args)

    if not filtered:
        return jsonify({"message": "No stat lines found matching the given filters."}), 404

    if filtered and 'error' in filtered[0]:
        return jsonify(filtered[0]), 400

    sorted_lines = sort_stat_lines(filtered, request.args)
    if sorted_lines and 'error' in sorted_lines[0]:
        return jsonify(sorted_lines[0]), 400

    return jsonify(sorted_lines)


@app.route('/giannis/career-averages')
def get_career_averages():
    error_response = check_data_ready('stat_lines')
    if error_response:
        return error_response

    total_games = len(data['stat_lines'])

    totals = {stat: 0 for stat in STAT_CATEGORIES}
    career_highs = {stat: 0 for stat in STAT_CATEGORIES}

    for line in data['stat_lines']:
        stats = line.get('stats', {})
        for stat in STAT_CATEGORIES:
            value = stats.get(stat, 0)
            totals[stat] += value
            if value > career_highs[stat]:
                career_highs[stat] = value

    averages = {stat: round(totals[stat] / total_games, 2) for stat in STAT_CATEGORIES}

    return jsonify({
        "total_games_analyzed": total_games,
        "averages": averages,
        "career_highs": career_highs,
        "source": "Calculated from curated dataset."
    })


@app.route('/giannis/stats-by-opponent')
def get_stats_by_opponent():
    error_response = check_data_ready('stat_lines')
    if error_response:
        return error_response

    opponent_name = request.args.get('opponent')

    if not opponent_name:
        return jsonify({"error": "Missing required query parameter: opponent."}), 400

    opponent_lower = opponent_name.lower()

    matched_stat_lines = [
        line for line in data['stat_lines']
        if opponent_lower in line.get('opponent', '').lower()
    ]

    if not matched_stat_lines:
        return jsonify({"message": f"No games found against {opponent_name}."}), 404

    return jsonify({
        "opponent_query": opponent_name,
        "matched_games": matched_stat_lines,
        "count": len(matched_stat_lines)
    })


@app.route('/giannis/doubles')
def get_doubles():
    error_response = check_data_ready('stat_lines')
    if error_response:
        return error_response

    type_filter = request.args.get('type', 'dd').lower()

    if type_filter not in ['dd', 'td']:
        return jsonify({"error": "Invalid type filter. Must be 'dd' (Double-Double) or 'td' (Triple-Double)."}), 400

    is_td_query = (type_filter == 'td')

    filtered_games = []

    for line in data['stat_lines']:
        stats = line.get('stats', {})

        if is_td_query and is_triple_double(stats):
            filtered_games.append(line)
        elif not is_td_query and is_double_double(stats):

            if not is_triple_double(stats) or is_td_query:
                filtered_games.append(line)

    if not filtered_games:
        return jsonify({"message": f"No {type_filter.upper()}s found in the dataset."}), 404

    return jsonify({
        "double_type": "Triple-Double" if is_td_query else "Double-Double",
        "matched_games": filtered_games,
        "count": len(filtered_games)
    })


@app.route('/giannis/dunks-by-type')
def get_dunks_by_type():
    error_response = check_data_ready('dunks_by_type')
    if error_response:
        return error_response

    return jsonify(data['dunks_by_type'])


@app.route('/giannis/dunks/count')
def get_dunk_count():
    error_response = check_data_ready('dunks_by_type')
    if error_response:
        return error_response

    return jsonify({
        "dunk_type_count": len(data['dunks_by_type'])
    })
@app.route('/bucks/championship-quotes')
def get_championship_quotes():
    error_response = check_data_ready('championship_quotes')
    if error_response:
        return error_response

    return jsonify(data['championship_quotes'])

@app.route('/giannis/funny-quotes')
def get_funny_quotes():
    error_response = check_data_ready('funny_quotes')
    if error_response:
        return error_response

    return jsonify(data['funny_quotes'])

@app.route('/search/quotes')
def search_quotes():
    error_response = check_data_ready('funny_quotes')
    if error_response:
        return error_response

    query = request.args.get('query')
    source_filter = request.args.get('source', '').lower()

    if not query:
        return jsonify({"error": "Missing required query parameter: query."}), 400
    query_lower = query.lower()
    all_quotes = []
    if source_filter == 'funny':
        quotes_to_search = data.get('funny_quotes', [])
        source_name = "funny"
    elif source_filter == 'championship':
        quotes_to_search = data.get('championship_quotes', [])
        source_name = "championship"
    else:
        quotes_to_search = data.get('funny_quotes', []) + data.get('championship_quotes', [])
        source_name = "all"

    for quote in quotes_to_search:
        if query_lower in quote.get('quote', '').lower() or \
                query_lower in quote.get('context', '').lower():
            quote_with_source = quote.copy()
            if quote in data.get('championship_quotes', []):
                quote_with_source['source'] = 'Championship'
            else:
                quote_with_source['source'] = 'Funny'

            all_quotes.append(quote_with_source)

    return jsonify({
        "query": query,
        "source_searched": source_name,
        "results": all_quotes,
        "count": len(all_quotes)
    })


if __name__ == '__main__':
    app.run(debug=True)