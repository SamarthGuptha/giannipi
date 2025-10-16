import heapq, json
from datetime import datetime
from typing import Dict, List, Any, Tuple

MILESTONE_THRESHOLDS = {
    '50 Points': {'stat': 'points', 'threshold': 50},
    '40 Points': {'stat': 'points', 'threshold': 40},
    '20 Rebounds': {'stat': 'rebounds', 'threshold': 20},
    '15 Assists': {'stat': 'assists', 'threshold': 15},
    '5 Blocks': {'stat': 'blocks', 'threshold': 5},
    '5 Steals': {'stat': 'steals', 'threshold': 5}
}

def _build_milestone_graph(stat_lines: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    milestone_dates: Dict[str, str] = {}

    for game in stat_lines:
        game_date_str = game['date']
        stats = game.get('stats', {})

        for name, params in MILESTONE_THRESHOLDS.items():
            stat_value = stats.get(params['stat'], 0)
            if stat_value>=params['threshold']:
                if name not in milestone_dates or game_date_str < milestone_dates[name]:
                    milestone_dates[name] = game_date_str

    sorted_milestones = sorted(milestone_dates.items(), key=lambda item: item[1])

    graph: Dict[str, Dict[str, int]] = {name: {} for name, _ in sorted_milestones}

    for i in range(len(sorted_milestones)):
        start_name, start_date_str = sorted_milestones[i]
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')

        for j in range(len(sorted_milestones)):
            if i==j:
                continue

            end_name, end_date_str = sorted_milestones[j]
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

            cost_days = abs((end_date-start_date).days)

            graph[start_name][end_name] = cost_days

    return graph

def find_shortest_path(stat_lines: List[Dict[str, Any]], start_node_name: str, end_node_name: str) -> tuple[dict[
    Any, Any], str] | dict[Any, Any]:
    graph = _build_milestone_graph(stat_lines)

    if start_node_name not in graph:
        return{}, f"Start milestone '{start_node_name}' not found in dataset."
    if end_node_name not in graph:
        return {}, f"End milestone '{end_node_name}' not found in dataset."

    distances: Dict[str, float] = {node: float('inf') for node in graph}
    distances[start_node_name] = 0
    previous_nodes: Dict[str, str] = {}

    pq: List[Tuple[float, str]] = [(0, start_node_name)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_distance>distances[current_node]: continue

        for neighbour, weight in graph[current_node].items():
            distance = current_distance+weight

            if distance<distances[neighbour]:
                distances[neighbour] = distance
                previous_nodes[neighbour] = current_node
                heapq.heappush(pq, (distance, neighbour))

    path_days = distances.get(end_node_name, float('inf'))

    if path_days==float('inf'):
        return {}, f"No path found from {start_node_name} to {end_node_name}."

    path: List[str] = []
    current = end_node_name
    while current:
        path.insert(0, current)
        current = previous_nodes.get(current)
        if current == start_node_name:
            path.insert(0, start_node_name)
            break

    return {
        "start_milestone": start_node_name,
        "end_milestone": end_node_name,
        "shortest_time_days": int(path_days),
        "milestone_path": path,
        "milestone_options": list(MILESTONE_THRESHOLDS.keys()),
        "interpolation": "The shortest path represents the minimum total number of days required to link this sequence of statistical achievements."
    }, ""

if __name__ == '__main__':
    DATA_FILE = 'giannis_data.json'
    try:
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
        stat_lines = data.get('stat_lines', [])
    except FileNotFoundError:
        print(f"CRITICAL ERROR: {DATA_FILE} not found.")
        stat_lines = []


    print("Testing Dijkstra service...")
    result, error = find_shortest_path(stat_lines, "50 Points", "5 Blocks")

    if error: print(f"Error: {error}")
    elif result: print(json.dump(result, indent=2))
    else: print("test failed: result empty.")