import json, random, re
from flask import Flask, jsonify, request
from datetime import datetime
from operator import itemgetter
import numpy as np

from comparison_service import calculate_similarity, get_career_highs, STAT_CATEGORIES
from analysis_service import analyze_quotes
from impact_service import rank_games_by_impact
from speaker_service import analyze_speakers
from probability_service import analyze_win_probability
from dijkstra_service import find_shortest_path, MILESTONE_THRESHOLDS
from streak_service import analyze_game_streaks
app = Flask(__name__)

DATA_FILE = 'giannis_data.json'

outcome_stats_cache = {}

VALID_STATS = {'points', 'rebounds', 'assists', 'steals', 'blocks'}
STAT_MAP = {'p': 'points', 'r': 'rebounds', 'a': 'assists', 's': 'steals', 'b': 'blocks'}
TRIVIA_RECIPES = [
    {
        "template": "Against which team did Giannis score {stats[points]} points on {date}?",
        "answer_key": "opponent",
        "distractor_key": "opponent"
    },
    {
        "template": "How many {stat_name} did Giannis have in his {stats[points]}-point game against the {opponent}?",
        "answer_key": "stats",
        "answer_subkey_pool": ["rebounds", "assists", "steals", "blocks"],
        "distractor_key": "stats"
    },
    {
        "template": "On what date did Giannis have his iconic '{fun_fact_short}' performance against the {opponent}?",
        "answer_key": "date",
        "distractor_key": "date"
    },
    {
        "template": "In the famous 'Game Ball Incident' game, Giannis set a franchise record with how many points?",
        "answer_key": "stats",
        "answer_sub_key": "points",
        "distractor_key": "stats",
        "requires_fun_fact_containing": "Game Ball Incident"
    }
]


try:
    global data
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


def get_distractors(stat_lines, correct_value, key, sub_key=None, num=3):
    distractor_pool = set()
    for game in stat_lines:
        if sub_key:
            value = game.get(key, {}).get(sub_key)
        else:
            value = game.get(key)

        if value is not None and value != correct_value: distractor_pool.add(value)

    if len(distractor_pool) < num: return []

    return random.sample(list(distractor_pool), num)

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

def execute_quote_search(quotes_to_search, query_lower, speaker_filter):
    all_quotes = []

    for quote_item in quotes_to_search:
        quote_lower = quote_item.get('quote', '').lower()
        context_lower = quote_item.get('context', '').lower()
        speaker_lower = quote_item.get('speaker', '').lower()

        is_speaker_match = (
            not speaker_filter or
            speaker_filter in speaker_lower
        )

        is_query_match = True
        if query_lower:
            is_query_match = (
                query_lower in quote_lower or
                query_lower in context_lower
            )

        if is_speaker_match and (not query_lower or is_query_match):
            quote_with_source = quote_item.copy()
            if quote_item in data.get('championship_quotes', []):
                quote_with_source['source'] = 'Championship'
            else: quote_with_source['source'] = 'Funny'

            all_quotes.append(quote_with_source)
    return all_quotes


def precompute_outcome_stats():
    global outcome_stats_cache
    if 'stat_lines' not in data:
        print("Could not pre-compute stats: 'stat_lines' not in data.")
        return

    win_totals = {'games_played': 0, 'points': 0, 'rebounds': 0, 'assists': 0, 'steals': 0, 'blocks': 0}
    loss_totals = {'games_played': 0, 'points': 0, 'rebounds': 0, 'assists': 0, 'steals': 0, 'blocks': 0}

    for game in data['stat_lines']:
        stats = game.get('stats', {})
        score = game.get('score', '')

        target_dict = None
        if score.startswith('W'):
            target_dict = win_totals
        elif score.startswith('L'):
            target_dict = loss_totals

        if target_dict is not None:
            target_dict['games_played'] += 1
            for stat_key in ['points', 'rebounds', 'assists', 'steals', 'blocks']:
                target_dict[stat_key] += stats.get(stat_key, 0)

    win_averages = {}
    if win_totals['games_played'] > 0:
        win_averages = {
            "games_played": win_totals['games_played'],
            "avg_points": round(win_totals['points'] / win_totals['games_played'], 1),
            "avg_rebounds": round(win_totals['rebounds'] / win_totals['games_played'], 1),
            "avg_assists": round(win_totals['assists'] / win_totals['games_played'], 1),
            "avg_steals": round(win_totals['steals'] / win_totals['games_played'], 1),
            "avg_blocks": round(win_totals['blocks'] / win_totals['games_played'], 1),
        }

    loss_averages = {}
    if loss_totals['games_played'] > 0:
        loss_averages = {
            "games_played": loss_totals['games_played'],
            "avg_points": round(loss_totals['points'] / loss_totals['games_played'], 1),
            "avg_rebounds": round(loss_totals['rebounds'] / loss_totals['games_played'], 1),
            "avg_assists": round(loss_totals['assists'] / loss_totals['games_played'], 1),
            "avg_steals": round(loss_totals['steals'] / loss_totals['games_played'], 1),
            "avg_blocks": round(loss_totals['blocks'] / loss_totals['games_played'], 1),
        }

    outcome_stats_cache = {"wins": win_averages, "losses": loss_averages}
    print("Successfully pre-computed and cached stats by outcome.")


def  calculate_impact_score(game_stats):
    return (
        game_stats.get('points', 0) +
        game_stats.get('rebounds', 0) +
        game_stats.get('assists', 0) +
        (game_stats.get('steals', 0) * 2) +
        (game_stats.get('blocks', 0) * 2)
    )

def parse_stat_string(stat_str):
    stats={}
    parts = stat_str.lower().split(',')

    for part in parts:
        part = part.strip()
        match = re.match(r'^(\d+)([prsab])$', part)
        if not match: return None, f"Invalid stat format: '{part}'. Use format like '50p, 10r, 5a'."

        value,key_char = match.groups()
        stat_name = STAT_MAP[key_char]
        stats[stat_name] = int(value)
    return stats, None

def calculate_percentage_difference(actual, hypothetical):
    if hypothetical == 0:
        return "N/A (cannot compare to 0)"

    diff = ((actual-hypothetical)/ hypothetical)*100

    return f"{diff:+.1f}%"

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
            "/analytics/simulate-game",
            "/analytics/time-gaps",
            "/analytics/stat-correlation",
            "/analytics/what-if"
            "/search/quotes?query=...&source=...&speaker=...",
            "/giannis/dunks-by-type",
            "/giannis/dunks/count",
            "/bucks/championship-quotes",
            "/giannis/funny-quotes",
            "/giannis/video-playlist",
            "/giannis/on-this-day",
            "/giannis/stats-by-outcome",
            "/giannis/opponent-deep-dive",
            "/trivia/generate"
        ]
    })

@app.route('/analytics/simulate-game')
def simulate_game():
    opponent_name = request.args.get('opponent')

    if not opponent_name:
        return jsonify({"error": "The 'opponent' query parameter is required."}), 400

    if 'stat_lines' not in data:
        return jsonify({"error": "The 'stat_lines' query parameter is required."}), 500

    historical_games = [
        game for game in data['stat_lines']
        if game.get('opponent', '').lower() == opponent_name.lower()
    ]

    if len(historical_games) < 2:
        return jsonify({
            "error": "Not enough historical data.",
            "message": f"Need atleast 2 games against {opponent_name} to run a simulation."
        }), 404


    stats_for_calc = {stat: [] for stat in STAT_CATEGORIES}
    bucks_scores = []
    opponent_scores = []

    for game in historical_games:
        for stat in stats_for_calc:
            stats_for_calc[stat].append(game.get('stats', {}).get(stat, 0))
        try:
            score_parts = game.get('score', '').split(' ')[1].split('-')
            bucks_score, opponent_score = int(score_parts[0]), int(score_parts[1])
            bucks_scores.append(bucks_score)
            opponent_scores.append(opponent_score)
        except (IndexError, ValueError):
            continue

    model = {"giannis_stats": {}, "scoring": {}}
    for stat, values in stats_for_calc.items():
        model["giannis_stats"][stat] = {"mean": np.mean(values), "std_dev": np.std(values)}

    model["scoring"]["bucks_mean"] = np.mean(bucks_scores)
    model["scoring"]["opponent_mean"] = np.mean(opponent_scores)
    model["scoring"]["opponent_std_dev"] = np.std(opponent_scores)

    simulated_stats = {}
    for stat, model_data in model["giannis_stats"].items():
        sim_value = np.random.normal(model_data["mean"], model_data["std_dev"])
        simulated_stats[stat]  = max(0, int(round(sim_value)))

    bucks_final_score = int(round(np.random.normal(model["scoring"]["bucks_mean"], 10)))
    opponent_final_score = int(round(np.random.normal(model["scoring"]["opponent_mean"], model["scoring"]["opponent_std_dev"])))

    response = {
        "simulation_details": {
            "opponent": opponent_name,
            "based_on_historical_games": len(historical_games)
        },
        "simulated_final_score": {
            "bucks": bucks_final_score,
            "opponent": opponent_final_score,
            "winner": "Milwaukee Bucks" if bucks_final_score>opponent_final_score else opponent_name
        },
        "giannis_simulated_stats": simulated_stats
    }

    return jsonify(response)

@app.route('/analytics/what-if')
def get_what_if_scenario():
    game_date_str = request.args.get('date')
    compare_to_str = request.args.get('compare_to')

    if not all([game_date_str, compare_to_str]):
        return jsonify({"error": "Missing required query params."}), 400

    try:
        datetime.strptime(game_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format."}), 400

    hypothetical_stats, error = parse_stat_string(compare_to_str)

    if error:
        return jsonify({"error": error}), 400

    if 'stat_lines' not in data:
        return jsonify({"error": "Stat lines unavailable"}), 500

    actual_game = next((game for game in data['stat_lines'] if game.get('date') == game_date_str), None)

    if not actual_game:
        return jsonify({"message": f"No game found in the dataset for the date '{game_date_str}'"}), 404

    actual_stats = actual_game.get('stats', {})
    percentage_difference = {}

    for stat, hypo_value in hypothetical_stats.items():
        actual_value = actual_stats.get(stat, 0)
        percentage_difference[stat] = calculate_percentage_difference(actual_value, hypo_value)

    points_hypo = hypothetical_stats.get('points')
    points_actual = actual_stats.get('points')

    narrative = f"In his performance against the {actual_game.get('opponent')}, Giannis scored {points_actual} points."
    if points_hypo is not None:
        points_diff = abs(points_actual - points_hypo)
        if points_actual<points_hypo:
            narrative += f" This was {points_diff} points shy of the hypothetical {points_hypo} points mark."
        else:
            narrative += f" This was {points_diff} points more than the hypothetical {points_hypo}-point mark."


    response = {
        "scenario": {
            "game_date": game_date_str,
            "opponent": actual_game.get('opponent'),
            "hypothetical_stats": hypothetical_stats,
        },
        "comparison": {
            "actual_stats": actual_stats,
            "percentage_difference": percentage_difference
        },
        "narrative": narrative
    }

    return jsonify(response)

@app.route('/trivia/generate')
def generate_trivia():
    if 'stat_lines' not in data or len(data['stat_lines'])<4:
        return jsonify({"error": "Not enough game data loaded to generate trivia."}), 500

    all_games = data['stat_lines']

    while True:
        recipe = random.choice(TRIVIA_RECIPES)
        answer_game = random.choice(all_games)

        if "requires_fun_fact_containing" in recipe:
            if recipe["requires_fun_fact_containing"] in answer_game.get("fun_fact", ""):
                break
            else: continue
        else: break

    distractor_games = [game for game in all_games if game != answer_game]

    template_data = answer_game.copy()
    template_data['fun_fact_short'] = template_data.get('fun_fact', '').split('.')[0]

    stat_to_ask = None
    if "answer_subkey_pool" in recipe:
        stat_to_ask = random.choice(recipe["answer_subkey_pool"])
        template_data['stat_name'] = stat_to_ask

    question = recipe["template"].format(**template_data)

    answer_subkey = recipe.get("answer_subkey") or stat_to_ask
    if answer_subkey:
        correct_answer = answer_game.get(recipe["answer_key"], {}).get(answer_subkey)
    else:
        correct_answer = answer_game.get(recipe["answer_key"])

    distractor_subkey = recipe.get("distractor_subkey") or answer_subkey
    incorrect_answers = get_distractors(
        distractor_games,
        correct_answer,
        recipe["distractor_key"],
        sub_key = distractor_subkey
    )

    if correct_answer is None or not incorrect_answers:
        return jsonify({"error": "Could not generate valid trivia question"}), 500


    return jsonify({
        "question": question,
        "correct_answer": correct_answer,
        "incorrect_answers": sorted(incorrect_answers)
    })

@app.route('/analytics/stat-correlation')
def get_stat_correlation():
    primary_stat = request.args.get('primary_stat')
    threshold_str = request.args.get('threshold')
    secondary_stat = request.args.get('secondary_stat')

    if not all([primary_stat, threshold_str, secondary_stat]): return jsonify({"error": "Missing required query params. please provide all three."})

    if primary_stat not in VALID_STATS: return jsonify({"error": f"Invalid 'primary_stat'. Must be one of {list(VALID_STATS)}."})
    if primary_stat not in VALID_STATS: return jsonify({"error": f"Invalid 'secondary_stat'. Must be one of {list(VALID_STATS)}."})

    try:
        threshold = int(threshold_str)
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid 'threshold'. Must be an integer."}), 400

    if 'stat_lines' not in data:
        return jsonify({"error": "Stat lines data unnavailable."}), 500

    filtered_games = [
        game for game in data['stat_lines']
        if game.get('stats', {}).get(primary_stat, 0) >= threshold
    ]

    total_secondary_stat = 0
    for game in filtered_games:
        total_secondary_stat += game.get('stats', {}).get(secondary_stat, 0)

    average_secondary_stat = round(total_secondary_stat / len(filtered_games), 1)

    response = {
        "query": {
            "primary_stat": primary_stat,
            "threshold": threshold,
            "secondary_stat_to_average": secondary_stat,
        },
        "analysis": {
            "matching_games_found": len(filtered_games),
            "average_value": average_secondary_stat,
        },
        "games_included_in_analysis": filtered_games
    }

    return jsonify(response)
@app.route('/giannis/opponent-deep-dive')
def get_opponent_deep_dive():
    opponent_name = request.args.get('opponent')

    if not opponent_name: return jsonify({"error": "The 'opponent' query parameter is required."})
    if 'stat_lines' not in data: return jsonify({"error": "The 'stat_lines' data unnavailable."})

    games_vs_opponent = [
        game for game in data['stat_lines']
        if game.get('opponent', '').lower() == opponent_name.lower()
    ]
    if not games_vs_opponent:
        return jsonify({"message": f"No games found in dataset against {opponent_name}."}), 404

    totals = {'points': 0, 'rebounds': 0, 'assists': 0, 'steals': 0, 'blocks': 0}

    for game in games_vs_opponent:
        for stat in totals:
            totals[stat] += game.get('stats', {}).get(stat, 0)

    num_games = len(games_vs_opponent)
    career_averages = {f"avg_{stat}": round(total / num_games, 1) for stat, total in totals.items()}

    career_highs = {'points': 0, 'rebounds': 0, 'assists': 0, 'steals': 0, 'blocks': 0}

    for game in games_vs_opponent:
        for stat in career_highs:
            current_stat = game.get('stats', {}).get(stat, 0)
            if current_stat>career_highs[stat]:
                career_highs[stat] = current_stat

    best_game  = None
    max_score = -1
    for game in games_vs_opponent:
        score = calculate_impact_score(game.get('stats', {}))
        if score>max_score:
            max_score=score
            best_game=game

    response = {
        "opponent": best_game.get('opponent'),
        "games_played_in_dataset": num_games,
        "career_averages": career_averages,
        "career_highs": career_highs,
        "best_game_details": best_game
    }

    return jsonify(response)




@app.route('/giannis/stats-by-outcome')
def get_stats_by_outcome():
    if not outcome_stats_cache:
        return jsonify({"error": "Data could not be processed"}), 500

    return jsonify(outcome_stats_cache)


@app.route('/giannis/video-playlist')
def get_video_playlist():
    if 'stat_lines' not in data:
        return jsonify({"error": "Stat Lines data not available or not loaded."}), 500

    stat_lines = data.get('stat_lines', [])

    sort_by = request.args.get('sort_by', 'date').lower()
    try:
        limit = int(request.args.get('limit', 10))
    except (ValueError, TypeError):
        return jsonify({"error": "Limit must be an integer."}), 500

    valid_sort_stats = ['points', 'rebounds', 'assists', 'steals', 'blocks']

    if sort_by == 'date':
        sorted_games = sorted(stat_lines, key=itemgetter('date'), reverse = True)
    elif sort_by in valid_sort_stats:
        sorted_games = sorted(
            stat_lines,
            key=lambda game: game.get('stats', {}).get(sort_by, 0),
            reverse = True
        )
    else:
        return jsonify({
            "error": f"invalid 'sort_by' parameter. Use 'date' or one of {valid_sort_stats}"
        }), 400

    limited_games = sorted_games[:limit]

    playlist = []
    for game in limited_games:
        if 'youtube_link' in game and game['youtube_link']:
            points = game.get('stats', {}).get('points', 'N/A')
            title = f"{points} points vs {game.get('opponent', 'N/A')} ({game.get('date', 'N/A')})"

            playlist.append({
                "title": title,
                "date": game.get('date'),
                "opponent": game.get('opponent'),
                "url": game.get('youtube_link')
            })
    return jsonify(playlist)


@app.route('/giannis/on-this-day')
def get_on_this_day():
    if 'stat_lines' not in data:
        return jsonify({"error": "Stat lines data not available."}), 500

    target_date_str = request.args.get('date')
    if target_date_str:
        try:
            datetime.strptime(f"2000-{target_date_str}", "%Y-%m-%d")
            target_md = target_date_str
        except ValueError:
            return jsonify({"error": "Invalid date format. Please use 'MM-DD'."}), 400
    else:
        today = datetime.now()
        target_md = today.strftime('%Y-%m-%d')

    stat_lines = data.get('stat_lines', [])
    found_games = []

    for game in stat_lines:
        game_date = game.get('date', '')
        if game_date[5:] == target_md:
            found_games.append(game)

    if not found_games:
        return jsonify({
            "message": f"No historical games found for the date {target_md}"
        }), 404

    return jsonify(found_games)


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

    all_quotes = execute_quote_search(quotes_to_search, query_lower, speaker_filter)

    if not all_quotes: return jsonify({"error": "No quotes found."}), 404

    return jsonify({
        "query": query,
        "source_searched": source_name,
        "speaker_filtered": speaker_filter if speaker_filter else "none",
        "results": all_quotes,
        "count": len(all_quotes)
    })

if __name__ == '__main__':
    app.run(debug=True)
