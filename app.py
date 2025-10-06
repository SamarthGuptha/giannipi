import json
import random
import os
from flask import Flask, jsonify, request
from functools import reduce
from datetime import datetime
DATA_FILE = "giannis_data.json"
giannis_stats_data = {}


def load_data():
    global giannis_stats_data
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, DATA_FILE)

        with open(file_path, 'r') as f:
            giannis_stats_data = json.load(f)
        print(f"Successfully loaded data from {DATA_FILE}")
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Ensure it is in the same directory as app.py.")
        giannis_stats_data = {}
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {DATA_FILE}. Check file integrity.")
        giannis_stats_data = {}
app = Flask(__name__)
load_data()

def filter_and_sort_stats(stat_lines, args):
    filtered_stats = stat_lines[:]
    filter_fields = {'points_min': 'points', 'rebounds_min': 'rebounds', 'assists_min': 'assists'}

    #datestuff
    date_start_str = args.get('date_start')
    date_end_str = args.get('date_end')
    DATE_FORMAT = "%Y-%m-%d"

    try:
        start_date = datetime.strptime(date_start_str, DATE_FORMAT) if date_start_str else datetime.min
        end_date = datetime.strptime(date_end_str, DATE_FORMAT) if date_end_str else datetime.max

        filtered_stats = [
            line for line in filtered_stats
            if start_date <= datetime.strptime(line['date'], DATE_FORMAT) <= end_date
        ]
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}, 400

    for arg_name, stat_key in filter_fields.items():
        min_value_str = args.get(arg_name)
        if min_value_str:
            try:
                min_value = int(min_value_str)
                filtered_stats = [
                    line for line in filtered_stats
                    if line['stats'].get(stat_key, 0) >= min_value
                ]
            except ValueError:
                print(f"Warning: Invalid integer value for {arg_name}: {min_value_str}")


    sort_by = args.get('sort_by', 'date')
    order = args.get('order', 'desc')

    if sort_by in ['points', 'rebounds', 'assists', 'steals', 'blocks']:
        sort_key = lambda line: line['stats'].get(sort_by, 0)
    else:
        sort_key = lambda line: line.get(sort_by)
    reverse = (order.lower() == 'desc')
    try:
        filtered_stats.sort(key=sort_key, reverse=reverse)
    except Exception as e:
        print(f"Warning: Failed to sort by {sort_by}. Error: {e}")

    return filtered_stats
# --- API Endpoints ---
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Giannis Signature Stats API!", "version": "2.0"})

@app.route('/giannis/stat-lines', methods=['GET'])
def get_stat_lines():
    stat_lines = giannis_stats_data.get('stat_lines', [])

    if not stat_lines:
        return jsonify({"error": "Stat lines data not available"}), 404
    if not request.args:
       return jsonify(random.choice(stat_lines))

    results = filter_and_sort_stats(stat_lines, request.args)

    if results:
        return jsonify(results)

    return jsonify({"message": "No stat lines found matching the criteria."}), 200

@app.route('/search/quotes', methods=['GET'])
def search_quotes():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "No query provided"}), 400

    search_term = query.lower()
    all_quotes = []

    championship = [dict(q, source="Championship") for q in giannis_stats_data.get('championship_quotes', [])]
    funny = [dict(q, source="Funny") for q in giannis_stats_data.get('funny_quotes', [])]

    all_quotes.extend(championship)
    all_quotes.extend(funny)

    results = []
    for item in all_quotes:
        quote_text = item.get('quote', '').lower()
        context_text = item.get('context', '').lower()

        if search_term in quote_text or search_term in context_text:
            results.append(item)

    return jsonify({
        "query": query,
        "count": len(results),
        "results": results
    })

@app.route('/giannis/dunks/count', methods=['GET'])
def get_dunk_count():
    dunks_data = giannis_stats_data.get('dunks_by_type', [])
    count = len(dunks_data)

    if count==0: return jsonify({"message": "No dunk types defined in dataset."}), 200

    return jsonify({
        "total_dunk_types": count,
        "summary": "This API tracks four distinct signature dunk types by Giannis Antetokounmpo.",
        "types_avilable": [d['type'] for d in dunks_data]
    }
    )
@app.route('/giannis/career-averages', methods=['GET'])
def get_career_averages():
    stat_lines = giannis_stats_data.get('stat_lines', [])

    if not stat_lines:
        return jsonify({"error": "Stat lines data not available for aggregation"}), 404

    total_games = len(stat_lines)

    totals = {
        "points": 0,
        "rebounds": 0,
        "assists": 0,
        "steals": 0,
        "blocks": 0
    }

    def calculate_totals(acc, line):
        stats = line.get('stats', {})
        for key in totals.keys():
            acc[key] += stats.get(key, 0)
        return acc

    total_stats = reduce(calculate_totals, stat_lines, totals)

    averages = {
        key: round(total_stats[key] / total_games, 2)
        for key in total_stats
    }
    career_high_points = max(
        [line['stats'].get('points', 0) for line in stat_lines],
        default=0
    )

    response_data = {
        "summary_title": f"Averages across {total_games} Signature Games",
        "total_games_analyzed": total_games,
        "averages": averages,
        "career_highs": {
            "points": career_high_points
        },
        "disclaimer": "These averages are based only on the signature stat lines provided in the dataset, not his actual career stats."
    }

    return jsonify(response_data)

@app.route('/giannis/dunks-by-type', methods=['GET'])
def get_dunks_by_type():
    dunks_data = giannis_stats_data.get('dunks_by_type', [])
    if dunks_data:
        return jsonify(dunks_data)

    return jsonify({"error": "Dunks by type data not available"}), 404


@app.route('/bucks/championship-quotes', methods=['GET'])
def get_championship_quotes():
    quotes_data = giannis_stats_data.get('championship_quotes', [])
    if quotes_data:
        return jsonify(quotes_data)

    return jsonify({"error": "Championship quotes data not available"}), 404

@app.route('/giannis/funny-quotes', methods=['GET'])
def get_funny_quotes():
    quotes_data = giannis_stats_data.get('funny_quotes', [])
    if quotes_data:
        return jsonify(quotes_data)

    return jsonify({"error": "Funny quotes data not available"}), 404

if __name__ == '__main__':
    print("Starting Flask server for local testing...")
    app.run(debug=True)