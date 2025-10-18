import json
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta

STAT_CATEGORIES = ['points', 'rebounds', 'assists', 'steals', 'blocks']

def analyze_game_streaks(
        stat_lines: List[Dict[str, Any]],
        stat_key: str,
        min_value: int,
        min_length: int
) -> tuple[list[Any], str] | None:
    if stat_key not in STAT_CATEGORIES: return [], f"Invalid stat key: '{stat_key}'. Must be one of: {', '.join(STAT_CATEGORIES)}"

    try:
        sorted_games = sorted(
            stat_lines,
            key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d')
        )
    except ValueError: return [], "Error parsing date in stat lines. Ensure all dates are in YYYY-MM-DD."
    except KeyError: return [], "stat lines are missing the 'date' key."


    if not sorted_games:
        return [], "No game data available for streak analysis."

    streaks_found = []
    current_streak = []

    for i in range(len(sorted_games)):
        game = sorted_games[i]

        game_stat = game.get('stats', {}).get(stat_key, 0)

        if game_stat>=min_value:
            current_streak.append(game)
        else:
            if len(current_streak)>=min_length:

                streak_start_date = current_streak[0]['date']
                streak_end_date = current_streak[-1]['date']

                total_stat = sum(g['stats'].get(stat_key, 0) for g in current_streak)
                avg_stat = round(total_stat / len(current_streak), 2)


                streaks_found.append({
                    "stat_key": stat_key,
                    "min_value": min_value,
                    "length": len(current_streak),
                    "start_date": streak_start_date,
                    "end_date": streak_end_date,
                    "total_stat_during_streak": total_stat,
                    "average_stat_during_streak": avg_stat,
                    "games": [
                        {
                            "date": g['date'],
                            "opponent": g['opponent'],
                            "score": g['score'],
                            "stat_value": g['stats'].get(stat_key, 0)
                        }
                        for g in current_streak
                    ]
                })

            current_streak = []

    if len(current_streak)>=min_length:
        streak_start_date = current_streak[0]['date']
        streak_end_date = current_streak[-1]['date']
        total_stat = sum(g['stats'].get(stat_key, 0) for g in current_streak)
        avg_stat = round(total_stat / len(current_streak), 2)

        streaks_found.append({
            "stat_key": stat_key,
            "min_value": min_value,
            "length": len(current_streak),
            "start_date": streak_start_date,
            "end_date": streak_end_date,
            "total_stat_during_streak": total_stat,
            "average_stat_during_streak": avg_stat,
            "games": [
                {
                    "date": g['date'],
                    "opponent": g['opponent'],
                    "score": g['score'],
                    "stat_value": g['stats'].get(stat_key, 0)
                }
                for g in current_streak
            ]
        })

    return streaks_found, ""

if __name__ == '__main__':
    DATA_FILE = 'giannis_data.json'
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        test_data = data.get('stat_lines', [])
    except FileNotFoundError:
        print(f"CRITIAL ERROR: {DATA_FILE} not found, cannot test service locally.")
        test_data = []
    except json.JSONDecodeError:
        print(f"CRITICAL ERROR: Failed to decode JSON from {DATA_FILE}.")
        test_data = []

    print('Testing Streak analysis service')

    streaks, error = analyze_game_streaks(test_data, 'points', 40, 1)

    if error: print(f"Error: {error}")
    elif streaks:
        print("\n--- Streaks of 40+ points (Min length 1) ---")
        print(json.dumps(streaks, indent=2))
    else:
        print("\n--- Streaks of 40+ Points (Min Length 1) ---")
        print("Test run completed, no streaks found with current data.")

    streaks_rb, error_rb = analyze_game_streaks(test_data, 'rebounds', 10, 2)

    if error_rb:
        print(f"Error: {error_rb}")
    elif streaks_rb:
        print("\n--- Streaks of 10+ Rebounds (Min Length 2) ---")
        print(json.dumps(streaks_rb, indent=2))
    else:
        print("\n--- Streaks of 10+ Rebounds (Min Length 2) ---")
        print("Test run completed, no streaks found with current data.")

