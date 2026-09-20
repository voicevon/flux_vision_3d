import numpy as np
import cv2
import networkx as nx

def graph_based_tracing(mask: np.ndarray, min_branch_len: int = 20) -> list:
    """Trace vessel graph from binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary vessel mask (uint8) where foreground == 255.
    min_branch_len : int
        Minimum length of a branch to keep (in pixels).

    Returns
    -------
    centrelines : list of np.ndarray
        Each element is an (N, 2) array of (x, y) coordinates representing a centreline path.
    """
    # 8‑connectivity graph
    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.where(binary)
    G = nx.Graph()
    for y, x in zip(ys, xs):
        G.add_node((y, x))
        # check 8 neighbours
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < mask.shape[0] and 0 <= nx_ < mask.shape[1] and binary[ny, nx_]:
                    G.add_edge((y, x), (ny, nx_))
    # Identify endpoints (degree 1) as start/stop points
    endpoints = [n for n, d in G.degree() if d == 1]
    visited = set()
    centrelines = []
    for ep in endpoints:
        if ep in visited:
            continue
        # BFS to follow path until another endpoint or dead end
        path = [ep]
        current = ep
        prev = None
        while True:
            visited.add(current)
            neighbors = [n for n in G.neighbors(current) if n != prev]
            if not neighbors:
                break
            if len(neighbors) > 1:
                # branching point, stop current path
                break
            nxt = neighbors[0]
            path.append(nxt)
            prev, current = current, nxt
            if G.degree(nxt) == 1:
                # reached another endpoint
                visited.add(nxt)
                break
        if len(path) >= min_branch_len:
            pts = np.array([[p[1], p[0]] for p in path], dtype=np.float32)
            centrelines.append(pts)
    return centrelines
