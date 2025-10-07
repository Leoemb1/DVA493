def read_graph(filename):
    graph = {}
    with open(filename, 'r') as f:
        for _ in range(3):
            next(f)

        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue  # hoppa över felaktiga rader
            c1, c2, dist_str = parts[0], parts[1], int(parts[2])
            try:
                dist = int(dist_str)
            except ValueError:
                continue

            if c1 not in graph:
                graph[c1] = []
            if c2 not in graph:
                graph[c2] = []
            graph[c1].append((c2, dist))
            graph[c2].append((c1, dist))
    return graph

def shortest_distance_to_F(graph, destination='F'):
    dist = {city: float('inf') for city in graph}
    dist[destination] = 0
    previous = {city: None for city in graph}

    # Relaxera kanterna (max antal noder - 1 gånger)
    for _ in range(len(graph) - 1):
        updated = False
        for city in graph:
            for neighbor, cost in graph[city]:
                if dist[neighbor] + cost < dist[city]:
                    dist[city] = dist[neighbor] + cost
                    previous[city] = neighbor
                    updated = True
        if not updated:
            break  # inga förbättringar, avsluta tidigt
    return dist, previous

def shortest_path(city, previous, destination='F'):
    path = []
    current = city
    while current != destination:
        if current is None:
            return None  # ingen väg finns
        path.append(current)
        current = previous[current]
    path.append(destination)
    return path

# Huvudprogram

filename = "city 1.txt"  # Byt namn om din fil heter något annat
graph = read_graph(filename)

distances, previous = shortest_distance_to_F(graph, destination='F')

print("Kortaste avstånd till F från varje stad:")
for city in sorted(distances):
    print(f"{city}: {distances[city]}")

print("\nExempel på kortaste vägar till F:")
for city in sorted(graph):
    if city == 'F':
        continue
    path = shortest_path(city, previous, destination='F')
    if path:
        print(f"{city} -> F: {' -> '.join(path)}")
    else:
        print(f"{city} -> F: Ingen väg finns")
