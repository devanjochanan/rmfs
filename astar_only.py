import heapq

class Node:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.g = 0
        self.h = 0

    def __lt__(self, other):
        return (self.g + self.h) < (other.g + other.h)

def astar(grid, start, end):
    def heuristic(node, end):
        return abs(node.x - end.x) + abs(node.y - end.y)

    open_set = []
    closed_set = set()

    start_node = Node(start[0], start[1])
    end_node = Node(end[0], end[1])

    open_set.append(start_node)

    while open_set:
        current_node = heapq.heappop(open_set)

        if current_node.x == end_node.x and current_node.y == end_node.y:
            path = []
            while current_node:
                path.append((current_node.x, current_node.y))
                current_node = current_node.parent
            return path[::-1]

        closed_set.add((current_node.x, current_node.y))

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            x, y = current_node.x + dx, current_node.y + dy

            if (
                0 <= x < len(grid)
                and 0 <= y < len(grid[0])
                and grid[x][y] == 0
                and (x, y) not in closed_set
            ):
                neighbor = Node(x, y, current_node)
                neighbor.g = current_node.g + 1
                neighbor.h = heuristic(neighbor, end_node)

                if neighbor not in open_set:
                    heapq.heappush(open_set, neighbor)

    return None

# Example usage:
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
]

start = (0, 0)
end = (4, 3)

for r in grid:
    print(r)
print("====")
path = astar(grid, start, end)

if path:
    for x, y in path:
        grid[x][y] = 2

    for row in grid:
        print(row)
else:
    print("No path found.")
