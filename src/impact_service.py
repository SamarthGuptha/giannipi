from typing import List, Dict, Any, Tuple
STAT_CATEGORIES = ['points', 'rebounds', 'assists', 'steals', 'blocks']


def calculate_impact_score(stats: Dict[str, int], weights: Dict[str, int]) -> int:
    score = 0

    for stat in STAT_CATEGORIES:
        weight = weights.get(stat, 0)
        value = stats.get(stat, 0)
        score += (value * weight)

    return score


def parse_weights(weights_str: str) -> Tuple[Dict[str, int], str]:
    try:
        raw_weights = [int(w.strip()) for w in weights_str.split(',')]
    except ValueError:
        return {}, "Weights must be integers and comma-separated."

    if len(raw_weights) != len(STAT_CATEGORIES):
        return {}, f"Expected {len(STAT_CATEGORIES)} weights (P, R, A, S, B), but received {len(raw_weights)}."

    weights_dict = {
        STAT_CATEGORIES[i]: raw_weights[i]
        for i in range(len(STAT_CATEGORIES))
    }

    return weights_dict, ""


def rank_games_by_impact(stat_lines: List[Dict[str, Any]], weights_str: str) -> Tuple[List[Dict[str, Any]], str]:
    weights_dict, error = parse_weights(weights_str)
    if error:
        return [], error

    ranked_games = []

    for line in stat_lines:
        impact_score = calculate_impact_score(line.get('stats', {}), weights_dict)
        game_data = line.copy()
        game_data['impact_score'] = impact_score

        ranked_games.append(game_data)
    ranked_games.sort(key=lambda x: x['impact_score'], reverse=True)

    return ranked_games, ""

if __name__ == '__main__':
    print("Game Impact Analysis Service loaded successfully.")