import json, random, os
from flask import Flask, jsonify

DATA_FILE = 'giannis_data.json'
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
        print(f"Error: {DATA_FILE} not found.")
        giannis_stats_data = {}
    except json.JSONDecodeError:
        print(f"Error: failed to decode JSON.")
        giannis_stats_data = {}


app = Flask(__name__)
load_data()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to giannipi!", "version": "1.0"})

@app.route('/giannis/stat-line-of-the-night', methods=['GET'])
def get_stat_line():
    stat_lines = giannis_stats_data.get('stat_lines', [])
    if stat_lines:
        random_stat_line = random.choice(stat_lines)
        return jsonify(random_stat_line)

    return jsonify({"message": "No stat_lines found."}), 404
@app.route('/bucks/championship-quotes', methods=['GET'])
def get_championship_quotes():
    quotes_data = giannis_stats_data.get('championship_quotes', [])
    if quotes_data:
        return jsonify(quotes_data)

    return jsonify({"error": "championship quotes data, unnavailable"}), 404
@app.route('/giannis/funny-quotes', methods=['GET'])
def get_funny_quotes():
    quotes_data = giannis_stats_data.get('funny_quotes', [])
    if quotes_data:
        return jsonify(quotes_data)

    return jsonify({"error": "funny quotes data, unnavailable"}), 404

if __name__ == '__main__':
    print("Starting flask")
    app.run(debug=True)