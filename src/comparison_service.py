import math
from typing import List, Dict, Any

STAT_CATEGORIES = ['points', 'rebounds', 'assists', 'steals', 'blocks']


def get_career_highs(stat_lines: List[Dict[str, Any]]) -> Dict[str, int]:
    career_highs = {stat: 0 for stat in STAT_CATEGORIES}

    for line in stat_lines:
        stats = line.get('stats', {})
        for stat in STAT_CATEGORIES:
            value = stats.get(stat, 0)
            if value > career_highs[stat]:
                career_highs[stat] = value

    return career_highs


def calculate_similarity(game1_stats: Dict[str, int], game2_stats: Dict[str, int],
                         career_highs: Dict[str, int]) -> float:
    distance_sq = 0
    for stat in STAT_CATEGORIES:
        high = career_highs.get(stat) if career_highs.get(stat, 0) > 0 else 1

        norm1 = game1_stats.get(stat, 0) / high
        norm2 = game2_stats.get(stat, 0) / high
        distance_sq += (norm1 - norm2) ** 2
    return math.sqrt(distance_sq)


if __name__ == '__main__':
    print("Comparison Service loaded successfully.")
