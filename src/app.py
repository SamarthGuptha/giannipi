import json
import random
from flask import Flask, jsonify, request
from datetime import datetime

from comparison_service import calculate_similarity, get_career_highs, STAT_CATEGORIES
from analysis_service import analyze_quotes
from impact_service import rank_games_by_impact
from speaker_service import analyze_speakers
from probability_service import analyze_win_probability
from dijkstra_service import find_shortest_path, MILESTONE_THRESHOLDS
from streak_service import analyze_game_streaks
app = Flask(__name__)

DATA_FILE = 'giannis_data.json'
try:
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
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
        return jsonify({
                           "error": f"Critical Error: Data file '{DATA_FILE}' could not be loaded. Ensure the file exists next to app.py."}), 500

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
        if stats.get(stat, 0) >= 10:
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
        "message": "Welcome to giannipi!",
        "endpoints": [
            "/giannis/stat-lines",
            "/giannis/career-averages",
            "/giannis/stats-by-opponent?opponent=...",
            "/giannis/doubles?type=dd",
            "/giannis/fun-facts?tag=...",
            "/giannis/compare-games?date=YYYY-MM-DD&limit=N",
            "/giannis/impact-ranking?weights=P,R,A,S,B",
            "/giannis/win-probability?date=YYYY-MM-DD",
            "/giannis/shortest-path?start=...&end=...",
            "/analytics/game-streaks?stat=...&min=...&length=...",
            "/analytics/quote-source-distribution",
            "/analytics/speaker-analysis",
            "/search/quotes?query=...&source=...&speaker=...",
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
        if not data['stat_lines']:
            return jsonify({"message": "No stat lines found in data source."}), 404
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
    career_highs = get_career_highs(data['stat_lines'])

    for line in data['stat_lines']:
        stats = line.get('stats', {})
        for stat in STAT_CATEGORIES:
            value = stats.get(stat, 0)
            totals[stat] += value

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


@app.route('/giannis/fun-facts')
def get_fun_facts():
    error_response = check_data_ready('stat_lines')
    if error_response:
        return error_response

    search_tag = request.args.get('tag', '').lower()

    fun_facts_list = []

    for line in data['stat_lines']:
        fact = line.get('fun_fact')

        if fact and search_tag in fact.lower():
            fun_facts_list.append({
                "date": line.get('date'),
                "opponent": line.get('opponent'),
                "score": line.get('score'),
                "fun_fact": fact,
                "youtube_link": line.get('youtube_link')
            })

    if not fun_facts_list and search_tag:
        return jsonify({"message": f"No fun facts found matching the tag: '{search_tag}'."}), 404

    return jsonify({
        "search_tag": search_tag if search_tag else "none",
        "results": fun_facts_list,
        "count": len(fun_facts_list)
    })


@app.route('/giannis/impact-ranking')
def get_impact_ranking():
    error_response = check_data_ready('stat_lines')
    if error_response:
        return error_response

    weights_str = request.args.get('weights')
    limit = request.args.get('limit', type=int, default=5)

    if not weights_str:
        return jsonify({
            "error": "Missing required query parameter: weights.",
            "format": "Weights must be provided as a comma-separated string for P, R, A, S, B (e.g., ?weights=2,3,1,5,4)."
        }), 400

    ranked_games, error = rank_games_by_impact(data['stat_lines'], weights_str)

    if error:
        return jsonify({"error": f"Invalid weights format: {error}"}), 400

    if not ranked_games:
        return jsonify({"message": "No games to rank."}), 404

    weights_dict = {
        'P': 0, 'R': 0, 'A': 0, 'S': 0, 'B': 0
    }
    try:
        raw_weights = [int(w.strip()) for w in weights_str.split(',')]
        if len(raw_weights) == 5:
            weights_dict = dict(zip(['P', 'R', 'A', 'S', 'B'], raw_weights))
    except Exception:
        pass

    return jsonify({
        "impact_weights": weights_dict,
        "ranking_criteria": "Points, Rebounds, Assists, Steals, Blocks",
        "top_ranked_games": ranked_games[:limit]
    })


@app.route('/giannis/win-probability')
def get_win_probability():
    error_response = check_data_ready('stat_lines')
    if error_response:
        return error_response

    target_date = request.args.get('date')

    if not target_date:
        return jsonify({"error": "Missing required query parameter: date (YYYY-MM-DD)."}), 400

    analysis_results, error = analyze_win_probability(data['stat_lines'], target_date)

    if error:
        return jsonify({"error": error}), 404 if "not found" in error else 400

    return jsonify(analysis_results)


@app.route('/giannis/shortest-path')
def get_shortest_path():
    error_response = check_data_ready('stat_lines')
    if error_response:
        return error_response

    start_node = request.args.get('start')
    end_node = request.args.get('end')

    if not start_node or not end_node:
        return jsonify({
            "error": "Missing required query parameters: start and end milestones.",
            "available_milestones": list(MILESTONE_THRESHOLDS.keys())
        }), 400

    path_results, error = find_shortest_path(data['stat_lines'], start_node, end_node)

    if error:
        return jsonify({"error": error}), 400

    return jsonify(path_results)


@app.route('/analytics/game-streaks')
def get_game_streaks():
    error_response = check_data_ready('stat_lines')
    if error_response:
        return error_response

    stat_key = request.args.get('stat')
    min_value = request.args.get('min', type=int)
    min_length = request.args.get('length', type=int)

    if not stat_key or min_value is None or min_length is None:
        return jsonify({
            "error": "Missing required query parameters.",
            "format": "Must provide 'stat' (e.g., points), 'min' (e.g., 30), and 'length' (e.g., 3)."
        }), 400

    streaks, error = analyze_game_streaks(
        data['stat_lines'],
        stat_key,
        min_value,
        min_length
    )

    if error:
        return jsonify({"error": error}), 400

    if not streaks:
        return jsonify({
                           "message": f"No streaks of {min_length} games or more found for {stat_key.capitalize()} >= {min_value}."}), 404

    return jsonify({
        "query": f"{min_length}+ games with {min_value}+ {stat_key.capitalize()}",
        "total_streaks_found": len(streaks),
        "streaks": streaks
    })


@app.route('/analytics/quote-source-distribution')
def get_quote_distribution_analysis():
    error_response = check_data_ready('championship_quotes')
    if error_response:
        error_response = check_data_ready('funny_quotes')
        if error_response:
            return error_response

    analysis_results = analyze_quotes(data)

    return jsonify(analysis_results)


@app.route('/analytics/speaker-analysis')
def get_speaker_analysis():
    error_response = check_data_ready('championship_quotes')
    if error_response:
        error_response = check_data_ready('funny_quotes')
        if error_response:
            return error_response

    analysis_results = analyze_speakers(data)

    return jsonify(analysis_results)


@app.route('/giannis/compare-games')
def compare_games():
    error_response = check_data_ready('stat_lines')
    if error_response:
        return error_response

    target_date = request.args.get('date')
    limit = request.args.get('limit', type=int, default=5)

    if not target_date:
        return jsonify({"error": "Missing required query parameter: date (YYYY-MM-DD)."}), 400

    try:
        target_game = next(
            (line for line in data['stat_lines'] if line['date'] == target_date),
            None
        )
    except Exception:
        return jsonify({"error": "Data structure error when searching for target date."}), 500

    if not target_game:
        return jsonify({"error": f"Game not found for date: {target_date}."}), 404

    career_highs = get_career_highs(data['stat_lines'])
    target_stats = target_game['stats']

    comparisons = []

    for game in data['stat_lines']:
        if game['date'] == target_date:
            continue #Skip comparing the game to itself

        similarity_score = calculate_similarity(
            target_stats,
            game['stats'],
            career_highs
        )

        result = {
            "date": game['date'],
            "opponent": game['opponent'],
            "score": game['score'],
            "stats": game['stats'],
            "similarity_distance": round(similarity_score, 4)
        }
        comparisons.append(result)
    comparisons.sort(key=lambda x: x['similarity_distance'])

    return jsonify({
        "target_game": {
            "date": target_game['date'],
            "opponent": target_game['opponent'],
            "stats": target_stats
        },
        "most_similar_games": comparisons[:limit],
        "comparison_method": "Euclidean Distance (normalized by career highs)"
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
        error_response = check_data_ready('championship_quotes')
        if error_response:
            return error_response

    query = request.args.get('query')
    source_filter = request.args.get('source', '').lower()
    speaker_filter = request.args.get('speaker', '').lower()
    if not query and not speaker_filter:
        return jsonify({"error": "Missing required query parameter: Must provide 'query' OR 'speaker' to search."}), 400

    query_lower = query.lower() if query else ''

    source_name = "all"

    if source_filter == 'funny':
        quotes_to_search = data.get('funny_quotes', [])
        source_name = "funny"
    elif source_filter == 'championship':
        quotes_to_search = data.get('championship_quotes', [])
        source_name = "championship"
    else:
        quotes_to_search = data.get('funny_quotes', []) + data.get('championship_quotes', [])

    all_quotes = []

    for quote in quotes_to_search:
        is_query_match = (
                query_lower in quote.get('quote', '').lower() or
                query_lower in quote.get('context', '').lower()
        )
        is_speaker_match = (
                not speaker_filter or
                speaker_filter in quote.get('speaker', '').lower()
        )

        if (not query or is_query_match) and is_speaker_match:
            quote_with_source = quote.copy()
            if quote in data.get('championship_quotes', []):
                quote_with_source['source'] = 'Championship'
            else:
                quote_with_source['source'] = 'Funny'

            all_quotes.append(quote_with_source)

    if not all_quotes:
        return jsonify({"message": "No quotes found matching the specified filters."}), 404

    return jsonify({
        "query": query,
        "source_searched": source_name,
        "speaker_filtered": speaker_filter if speaker_filter else "none",
        "results": all_quotes,
        "count": len(all_quotes)
    })


if __name__ == '__main__':
    app.run(debug=True)
