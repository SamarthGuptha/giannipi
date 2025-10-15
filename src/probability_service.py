import math
from typing import Dict, List, Any, Tuple

STAT_CATEGORIES = ['points', 'rebounds', 'assists', 'steals', 'blocks']

def _get_averages(games: List[Dict[str, Any]]) -> Dict[str, float]:
    if not games: return {stat: 0.0 for stat in STAT_CATEGORIES}

    totals = {stat: 0 for stat in STAT_CATEGORIES}

    for game in games:
        stats = game.get('stats', {})
        for stat in STAT_CATEGORIES:
            totals[stat]+=stats.get(stat, 0)

    num_games = len(games)
    return {stat: round(totals[stat]/num_games, 2) for stat in STAT_CATEGORIES}

def _euclidean_distance(target_stats: Dict[str, int], avg_stats: Dict[str, float]) -> float:
    sum_of_squares = 0

    for stat in STAT_CATEGORIES:
        target_val = target_stats.get(stat, 0)
        avg_val = avg_stats.get(stat, 0.0)
        sum_of_squares+=(target_val - avg_val)**2

    return math.sqrt(sum_of_squares)

def analyze_win_probability(stat_lines: List[Dict[str, Any]], target_date: str) -> tuple[dict[Any, Any], str] | tuple[
    dict[str, dict[str, str]]]:
    wins = [game for game in stat_lines if game.get('score', '').startswith('W')]
    losses = [game for game in stat_lines if game.get('score', ''). startswith('L')]

    if not wins or not losses:
        return {}, "Insufficient data: Need both winning and losing games to establish baselines."

    target_game = next(
        (game for game in stat_lines if game['date']==target_date),
        None
    )
    if not target_game:
        return {}, f"Target game not found for date: {target_date}."

    target_stats = target_game.get('stats', {})

    avg_win_stats = _get_averages(wins)
    avg_loss_stats = _get_averages(losses)

    dist_to_win = _euclidean_distance(target_stats, avg_win_stats)
    dist_to_loss = _euclidean_distance(target_stats, avg_loss_stats)

    win_probability_score = round(dist_to_win, 4)
    loss_probability_score = round(dist_to_loss, 4)

    if win_probability_score < loss_probability_score:
        prediction = "likely Win performance"
        is_correct = target_game.get('score', '').startswith('W')
    elif loss_probability_score < win_probability_score:
        prediction = "Likely loss performance"
        is_correct = target_game.get('score', '').startswith('L')
    else:
        prediction = "Neutral Performance"
        is_correct = None

    return {
        "target_game": {
            "date": target_date,
            "opponent": target_game.get('opponent'),
            "actual_outcome": target_game.get('score'),
            "stats": target_stats
        },
        "analysis_baselines": {
            "win_average_stats": avg_win_stats,
            "loss_average_stats": avg_loss_stats,
            "total_wins": len(wins),
            "total_losses": len(losses)
        },
        "probability_scores": {
            "distance_to_win_average": win_probability_score,
            "distance_to_loss_average": loss_probability_score,
            "predicted_performance": prediction,
            "prediction_correct": is_correct
        },
        "score_interpretation": "Lower distance score indicates higher statistical similarity"
    }, ""

if __name__ == '__main__':
    print("Victory Probability Analysis Service loaded.")