"""2D path planning for the base: occupancy grid, obstacle inflation and A*.

The robot is treated as a point on a grid where every obstacle has been grown by the robot's size.
ROBOT_RADIUS_M covers the stowed arm while turning on the spot: the body reaches 0.5 m from the base
origin, the stowed arm 0.73 m forward, measured from the link bounding boxes.
"""
import heapq

import numpy as np
from scipy import ndimage

RESOLUTION_M = 0.05
HALF_SIZE_M = 4.5
ROBOT_RADIUS_M = 0.80
BODY_RADIUS_M = 0.62      # the robot's true reach around its rotation centre: for the last stretch into the parking spot


class Grid:
    """Square occupancy grid centred on `centre` (world x, y). Cells are True when occupied."""

    def __init__(self, centre=(0.0, 0.0), half_size=HALF_SIZE_M, resolution=RESOLUTION_M):
        self.centre = np.asarray(centre, float)
        self.res = resolution
        self.n = int(round(2 * half_size / resolution))
        self.origin = self.centre - half_size
        self.occ = np.zeros((self.n, self.n), bool)

    def to_cell(self, xy):
        return tuple(np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(int))

    def to_world(self, cell):
        return self.origin + (np.asarray(cell, float) + 0.5) * self.res

    def inside(self, cell):
        return 0 <= cell[0] < self.n and 0 <= cell[1] < self.n

    def cell_centres(self):
        axis = self.origin[0] + (np.arange(self.n) + 0.5) * self.res
        axis_y = self.origin[1] + (np.arange(self.n) + 0.5) * self.res
        return np.meshgrid(axis, axis_y, indexing="ij")

    def inflated(self, radius=ROBOT_RADIUS_M):
        """Cells the robot's centre must stay out of."""
        if not self.occ.any():
            return np.zeros_like(self.occ)
        return ndimage.distance_transform_edt(~self.occ) * self.res < radius


def rasterise(grid, obstacles):
    """Mark true obstacle shapes on the grid (used for scene checks, not by the driving robot)."""
    X, Y = grid.cell_centres()
    for o in obstacles:
        dx, dy = X - o["xy"][0], Y - o["xy"][1]
        if o["shape"] == "cylinder":
            grid.occ |= dx ** 2 + dy ** 2 <= o["radius"] ** 2
        else:
            c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
            u, v = c * dx + s * dy, -s * dx + c * dy
            grid.occ |= (np.abs(u) <= o["half_extents"][0]) & (np.abs(v) <= o["half_extents"][1])
    return grid


def astar(blocked, start, goal):
    """8-connected A* on a boolean grid. Returns a list of cells from start to goal, or None."""
    if blocked[goal]:
        return None
    n = blocked.shape[0]
    moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 2 ** 0.5), (1, -1, 2 ** 0.5), (-1, 1, 2 ** 0.5), (-1, -1, 2 ** 0.5)]
    def h(c):
        dx, dy = abs(c[0] - goal[0]), abs(c[1] - goal[1])
        return max(dx, dy) + (2 ** 0.5 - 1) * min(dx, dy)
    cost, parent, frontier = {start: 0.0}, {}, [(h(start), start)]
    closed = set()
    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur == goal:
            path = [cur]
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            return path[::-1]
        if cur in closed:
            continue
        closed.add(cur)
        for dx, dy, step in moves:
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < n and 0 <= nxt[1] < n) or blocked[nxt]:
                continue
            if dx and dy and (blocked[cur[0] + dx, cur[1]] or blocked[cur[0], cur[1] + dy]):
                continue                                  # no cutting across a blocked corner
            new = cost[cur] + step
            if new < cost.get(nxt, np.inf):
                cost[nxt], parent[nxt] = new, cur
                heapq.heappush(frontier, (new + h(nxt), nxt))
    return None


def path_length(points):
    points = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0


def straighten(cells, blocked):
    """Shorten a grid path into straight segments: keep a corner only where a straight line would hit something."""
    def clear(a, b):
        steps = int(max(abs(a[0] - b[0]), abs(a[1] - b[1]))) * 2 + 1
        xs = np.linspace(a[0], b[0], steps).round().astype(int)
        ys = np.linspace(a[1], b[1], steps).round().astype(int)
        return not blocked[xs, ys].any()
    out, i = [cells[0]], 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not clear(cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def route_blocked(grid, points, from_index=1, skip_near=None, skip_m=0.3, radius=ROBOT_RADIUS_M):
    """Has a newly seen object landed on the remaining route? Ignores the part within skip_m of the robot, which may
    already be inside the safety margin while leaving it."""
    blocked = grid.inflated(radius)
    pts = np.asarray(points, float)
    if skip_near is not None:
        pts = np.vstack([skip_near, pts[from_index:]])
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(np.linalg.norm(b - a) / grid.res) + 2
        for q in a + np.linspace(0, 1, n)[:, None] * (b - a):
            if skip_near is not None and np.linalg.norm(q - skip_near) < skip_m:
                continue
            c = grid.to_cell(q)
            if grid.inside(c) and blocked[c]:
                return True
    return False


def plan_path(grid, start_xy, goal_xy, radius=ROBOT_RADIUS_M, goal_margin=None):
    """World-frame path from start to goal, or None. A start inside the inflated zone (the robot has
    drifted close to something) is allowed: the start cell is freed so the robot can leave."""
    blocked = grid.inflated(radius)
    s, g = grid.to_cell(start_xy), grid.to_cell(goal_xy)
    if not grid.inside(s) or not grid.inside(g):
        return None
    i0, j0 = np.ogrid[:grid.n, :grid.n]
    if blocked[s]:
        blocked = blocked & ~((i0 - s[0]) ** 2 + (j0 - s[1]) ** 2 <= (radius / grid.res) ** 2)
    if blocked[g] and goal_margin is not None:   # the goal may sit nearer an object, within goal_margin at the least
        near_goal = (i0 - g[0]) ** 2 + (j0 - g[1]) ** 2 <= (radius / grid.res) ** 2
        blocked = (blocked & ~near_goal) | grid.inflated(goal_margin)
    cells = astar(blocked, s, g)
    if cells is None:
        return None
    # The first and last points are the exact start and goal, not their cell centres
    points = [grid.to_world(c) for c in straighten(cells, blocked)]
    points[0], points[-1] = np.asarray(start_xy, float), np.asarray(goal_xy, float)
    return points
