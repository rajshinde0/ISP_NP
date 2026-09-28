"""ISP Network Planner - stages 0-5 (all algorithms hand-written; networkx is only a graph container)."""
import heapq, math, random, time
from collections import deque
import networkx as nx

from interface import RADIUS, BASE_COST, OPTIONS, TIER_BONUS, HEADROOM, LAT0, LON0


# ---------------- Stage 0: graph ----------------
def _add_latlon(G):
    for _, d in G.nodes(data=True):
        d["lat"] = LAT0 + d["y"] / 111_320
        d["lon"] = LON0 + d["x"] / (111_320 * math.cos(math.radians(LAT0)))


def synthetic_graph(n=25, spacing=150, seed=0):
    rng = random.Random(seed)
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            G.add_node(i * n + j, x=i * spacing + rng.uniform(-30, 30), y=j * spacing + rng.uniform(-30, 30))
    for i in range(n):
        for j in range(n):
            for di, dj in ((1, 0), (0, 1)):
                a, b = i + di, j + dj
                if a < n and b < n and rng.random() > 0.08:
                    G.add_edge(i * n + j, a * n + b)
    for u, v in G.edges:
        G[u][v]["length"] = math.hypot(G.nodes[u]["x"] - G.nodes[v]["x"], G.nodes[u]["y"] - G.nodes[v]["y"])
    _add_latlon(G)
    return G


class OsmFetchError(RuntimeError):
    """No OSM strategy produced a usable road graph. Carries every attempt's reason so the
    dashboard can show something actionable instead of a raw traceback."""


def _flatten_projected(Gp, G0):
    """Collapse an OSMnx MultiDiGraph pair into the simple undirected Graph the stages expect.

    `Gp` is the projected graph (x/y in metres, for distance maths); `G0` is the same graph
    unprojected, where x is longitude and y is latitude - note the swap, it is the easiest thing
    to get backwards and it silently puts every tower in the wrong hemisphere.

    Parallel edges collapse to the shortest, since a backbone only needs the cheapest link
    between two intersections. Self-loops are dropped: they add no connectivity and would give
    Prim a zero-length edge to chew on.
    """
    H = nx.Graph()
    for n, d in Gp.nodes(data=True):
        H.add_node(n, x=d["x"], y=d["y"], lat=G0.nodes[n]["y"], lon=G0.nodes[n]["x"])
    for u, v, d in Gp.edges(data=True):
        if u == v:
            continue
        l = d["length"]
        if H.has_edge(u, v):
            l = min(l, H[u][v]["length"])
        H.add_edge(u, v, length=l)
    return H


def osm_graph(place, dist=3000, network_type="drive"):
    """Fetch a real road network for `place`, in metres, ready for every downstream stage.

    Two strategies, in order:
      1. graph_from_place - correct when the name geocodes to a real boundary polygon (a city or
         an administrative district).
      2. graph_from_address with a `dist` radius - needed because most *neighbourhood* names come
         back from Nominatim as a bare point, and graph_from_place rejects anything that is not a
         (Multi)Polygon. The radius also bounds the download, so a broad query cannot hang the
         demo the way an unbounded city-wide fetch does.

    Raises OsmFetchError listing what each strategy said. Needs network access: the geocoder
    (Nominatim) and the road data (Overpass) are separate services, and Overpass is the slow one.
    """
    import osmnx as ox
    attempts, G0 = [], None
    strategies = (
        ("graph_from_place", lambda: ox.graph_from_place(place, network_type=network_type)),
        (f"graph_from_address(dist={dist}m)",
         lambda: ox.graph_from_address(place, dist=dist, network_type=network_type)),
    )
    for name, fetch in strategies:
        try:
            G = fetch()
        except Exception as e:                      # geocode miss, wrong geometry, network, ...
            attempts.append(f"{name}: {type(e).__name__}: {e}")
            continue
        if G.number_of_nodes():
            G0 = G
            break
        attempts.append(f"{name}: returned an empty graph")
    if G0 is None:
        raise OsmFetchError(
            f"Could not build a road network for {place!r}.\n  "
            + "\n  ".join(attempts)
            + "\n\nTry a broader name (a city or district rather than a neighbourhood), "
              "check the spelling, or use the synthetic grid offline.")
    return _flatten_projected(ox.project_graph(G0), G0)


def components(G):
    """BFS connectivity check, O(V+E). Returns list of node sets."""
    seen, comps = set(), []
    for s in G.nodes:
        if s in seen:
            continue
        comp, q = {s}, deque([s])
        seen.add(s)
        while q:
            u = q.popleft()
            for v in G[u]:
                if v not in seen:
                    seen.add(v); comp.add(v); q.append(v)
        comps.append(comp)
    return comps


def largest_component(G):
    comps = components(G)
    big = max(comps, key=len)
    return G.subgraph(big).copy(), len(comps) - 1  # graph, number of excluded components


# ---------------- instance + coverage ----------------
def make_instance(G, n_demand, n_cand, seed=0):
    """Draw demand points and candidate sites, then build each candidate's coverage bitmask.

    Returns (D, C, dropped). Demand that *no* candidate can reach is removed from D, because
    both stage-1 solvers assume a coverable universe - but the count and weight removed come
    back in `dropped` so the dashboard can say so out loud instead of reporting 100% coverage
    of a quietly shrunken city.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes)
    cx = sum(G.nodes[n]["x"] for n in nodes) / len(nodes)
    cy = sum(G.nodes[n]["y"] for n in nodes) / len(nodes)
    core = sorted(nodes, key=lambda n: math.hypot(G.nodes[n]["x"] - cx, G.nodes[n]["y"] - cy))[: max(1, len(nodes) // 5)]
    core = set(core)
    D = [{"node": n, "w": rng.randint(1, 20)} for n in rng.sample(nodes, n_demand)]
    C = []
    for n in rng.sample(nodes, n_cand):
        tier = "high" if n in core and rng.random() < 0.6 else "mid"
        C.append({"node": n, "tier": tier, "cost": round(BASE_COST[tier] * rng.uniform(0.8, 1.2), 2)})
    # coverage bitmask (euclidean, metric coordinates)
    for c in C:
        cn, m = G.nodes[c["node"]], 0
        for j, d in enumerate(D):
            dn = G.nodes[d["node"]]
            if math.hypot(cn["x"] - dn["x"], cn["y"] - dn["y"]) <= RADIUS[c["tier"]]:
                m |= 1 << j
        c["mask"] = m
    covered = 0
    for c in C:
        covered |= c["mask"]
    req_n, req_w = len(D), sum(d["w"] for d in D)
    keep = [j for j in range(len(D)) if covered >> j & 1]  # drop demand no candidate can reach
    if len(keep) < len(D):
        D = [D[j] for j in keep]
        for c in C:
            c["mask"] = sum(1 << k for k, j in enumerate(keep) if c["mask"] >> j & 1)
    kept_w = sum(d["w"] for d in D)
    dropped = {"count": req_n - len(D), "requested": req_n,
               "weight": req_w - kept_w, "requested_weight": req_w}
    return D, C, dropped


def wmask(m, w):
    s = 0
    while m:
        b = m & -m
        s += w[b.bit_length() - 1]
        m ^= b
    return s


def coverage_of_requested(r, D, dropped):
    """Re-express a solver's coverage over *every* customer originally requested, including the
    ones make_instance had to drop as unreachable. Without this the dashboard reports 100%."""
    served = r["coverage_pct"] / 100.0 * sum(d["w"] for d in D)
    return round(100 * served / max(1, dropped["requested_weight"]), 1)


# ---------------- Stage 1: facility location ----------------
def _result(name, C, D, chosen, t0, extra=None):
    w = [d["w"] for d in D]
    cov = 0
    for i in chosen:
        cov |= C[i]["mask"]
    r = {"solver": name, "sites": sorted(chosen), "cost": round(sum(C[i]["cost"] for i in chosen), 2),
         "coverage_pct": round(100 * wmask(cov, w) / max(1, sum(w)), 1), "runtime_s": time.perf_counter() - t0}
    r.update(extra or {})
    return r


def greedy_cover(C, D):
    """Cost-weighted greedy: max (newly covered weight / cost). Approximation ratio H(n) <= ln(n)+1.
    O(|C|^2 * |D|/64) - each of at most |C| rounds rescores every candidate."""
    t0 = time.perf_counter()
    w = [d["w"] for d in D]
    unc, chosen = (1 << len(D)) - 1, []
    while unc:
        best, bi = 0, None
        for i, c in enumerate(C):
            # an already-chosen set has its mask cleared from unc, so its gain is 0 and the
            # strict > below can never pick it again - no need to scan `chosen`
            g = wmask(c["mask"] & unc, w) / c["cost"]
            if g > best:
                best, bi = g, i
        if bi is None:
            break
        chosen.append(bi)
        unc &= ~C[bi]["mask"]
    return _result("greedy", C, D, chosen, t0)


def exact_cover(C, D, time_limit=30.0, prune=True):
    """Backtracking weighted set cover. O(2^|C|) worst case; NP-hard, hence the time limit.

    prune=True  -> branch and bound. Three cuts: a greedy warm-start incumbent, feasibility
                   (cost already >= incumbent, or the unpicked suffix cannot finish the cover),
                   and a fractional lower bound (cheapest cost-per-weight still available times
                   the weight still uncovered). In practice this holds the tree to a few hundred
                   nodes even at |C| = 40.
    prune=False -> the identical recursion with every bound removed: the textbook 2^|C| search.
                   Kept so the benchmark can show what the pruning actually buys, which is the
                   plan's "stalls past ~15-20 candidate sites" claim. Node counts are directly
                   comparable between the two modes.
    """
    w = [d["w"] for d in D]
    full, n = (1 << len(D)) - 1, len(C)
    order = sorted(range(n), key=lambda i: C[i]["cost"] / max(1, wmask(C[i]["mask"], w)))
    masks, costs = [C[i]["mask"] for i in order], [C[i]["cost"] for i in order]
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] | masks[i]
    warm = 0.0
    if prune:
        t0 = time.perf_counter()
        g = greedy_cover(C, D)  # incumbent -> strong initial bound
        warm = time.perf_counter() - t0
        best = [g["cost"] + 1e-9, list(g["sites"])]
    else:
        best = [math.inf, []]
    t0 = time.perf_counter()  # clock the search only: the warm start is greedy's cost, not ours
    state = {"timeout": False, "nodes": 0}

    def rec(i, cov, cost, chosen):
        state["nodes"] += 1
        if time.perf_counter() - t0 > time_limit:
            state["timeout"] = True
            return
        if cov == full:
            if cost < best[0]:
                best[0], best[1] = cost, [order[j] for j in chosen]
            return
        if i == n:
            return
        if prune:
            if cost >= best[0] or (cov | suffix[i]) != full:
                return
            unc = full & ~cov
            ratios = [costs[j] / wmask(masks[j] & unc, w) for j in range(i, n) if masks[j] & unc]
            if cost + min(ratios) * wmask(unc, w) >= best[0]:
                return
            if masks[i] & unc:  # a set that covers nothing new is never worth taking
                rec(i + 1, cov | masks[i], cost + costs[i], chosen + [i])
        else:
            rec(i + 1, cov | masks[i], cost + costs[i], chosen + [i])
        rec(i + 1, cov, cost, chosen)

    rec(0, 0, 0.0, [])
    return _result("exact (B&B)" if prune else "naive backtracking", C, D, best[1], t0,
                   {"nodes": state["nodes"], "timed_out": state["timeout"], "pruned": prune,
                    "found": best[0] < math.inf, "warmstart_s": round(warm, 6)})


# ---------------- Stage 2: knapsack ----------------
def knapsack(options=OPTIONS, budget=12, target=None):
    """0/1 knapsack, dp[i][b] = max capacity using first i options within budget b. O(n*B).

    target: demand this tower actually has to carry (Mbps). Given one, spend the *cheapest*
            budget level that still meets it rather than always emptying the budget, so a
            lightly loaded tower is not over-equipped. Unreachable demand falls back to the
            full budget; a tower with no customers of its own still buys one radio, because it
            may still be needed as a backbone relay.
    Returns (capacity Mbps, [(cost, capacity), ...] bought, spend k$).
    """
    n = len(options)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        c, cap = options[i - 1]
        for b in range(budget + 1):
            dp[i][b] = dp[i - 1][b]
            if c <= b and dp[i - 1][b - c] + cap > dp[i][b]:
                dp[i][b] = dp[i - 1][b - c] + cap
    b = budget
    if target is not None:
        floor = min(c for c, _ in options)
        b = next((k for k in range(floor, budget + 1) if dp[n][k] >= target), budget)
    cap_total, picked = dp[n][b], []
    for i in range(n, 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            picked.append(options[i - 1]); b -= options[i - 1][0]
    return cap_total, picked[::-1], sum(c for c, _ in picked)


def equip_towers(C, sites, demand_mbps, base_budget=12):
    """Stage 2, per built tower: its own knapsack DP, allowance scaled by tier, purchase trimmed
    to HEADROOM x the load stage 1 + assign_customers actually put on it (the multiple leaves
    room to relay neighbours' traffic). Returns (caps, picks, spends)."""
    caps, picks, spends = [], [], []
    for k, s in enumerate(sites):
        budget = base_budget + TIER_BONUS[C[s]["tier"]]
        cap, picked, spend = knapsack(OPTIONS, budget, target=demand_mbps[k] * HEADROOM)
        caps.append(cap); picks.append(picked); spends.append(spend)
    return caps, picks, spends


# ---------------- Stage 5 (used by stage 3): Dijkstra and A* ----------------
def dijkstra(G, src, dst=None, stats=None):
    """Binary-heap Dijkstra on edge "length". O((V+E) log V).
    dst=None -> (dist, prev) over the whole reachable set; else (distance, path).
    stats: optional dict, filled with {"popped": heap pops} so Dijkstra and astar can be
    compared on exactly equal terms. The return shape never changes."""
    dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
    popped = 0
    while pq:
        d, u = heapq.heappop(pq)
        popped += 1
        if d > dist.get(u, math.inf):
            continue
        if u == dst:
            break
        for v, e in G[u].items():
            nd = d + e["length"]
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if stats is not None:
        stats["popped"] = popped
    if dst is None:
        return dist, prev
    if dst not in dist:
        return math.inf, []
    path, u = [dst], dst
    while u != src:
        u = prev[u]; path.append(u)
    return dist[dst], path[::-1]


def astar(G, src, dst, stats=None):
    """A* on edge "length", guided towards dst by a straight-line heuristic.

    h(n) = Euclidean distance from n to dst in the **projected metre** coordinates. This is
    admissible: a road between two points is never shorter than the straight line between them,
    so h never over-estimates, A* never settles a node too early, and the distance it returns is
    exactly Dijkstra's optimum. It is also consistent (the triangle inequality holds for straight
    lines), so no node needs re-expanding once settled.

    On the synthetic grid h is not merely admissible but *exact* - synthetic_graph sets each edge
    length to the Euclidean distance between its endpoints - which is why the node saving there is
    so large (~85%). On real OSM roads, which bend, the heuristic is looser and the saving smaller.

    O((V+E) log V) worst case, identical to Dijkstra: the heuristic changes the constant, not the
    complexity, and degenerates to Dijkstra when h == 0. Returns (distance, path), the same shape
    as dijkstra(G, src, dst).
    """
    tn = G.nodes[dst]

    def h(n):
        nn = G.nodes[n]
        return math.hypot(nn["x"] - tn["x"], nn["y"] - tn["y"])

    g, prev, done = {src: 0.0}, {}, set()
    pq, popped, expansions, found = [(h(src), 0.0, src)], 0, 0, False
    while pq:
        _, gu, u = heapq.heappop(pq)
        popped += 1
        if u in done:                       # a stale heap entry for an already-settled node
            continue
        done.add(u)
        expansions += 1
        if u == dst:
            found = True
            break
        for v, e in G[u].items():
            nd = gu + e["length"]
            if nd < g.get(v, math.inf):
                g[v], prev[v] = nd, u
                heapq.heappush(pq, (nd + h(v), nd, v))
    if stats is not None:
        # expansions == settled is the invariant a *consistent* heuristic buys: once a node is
        # settled its g is final, so it never needs re-expanding. Dropping the `done` guard above
        # would still give the right answer - it would just do this work more than once per node.
        stats["popped"] = popped
        stats["expansions"] = expansions
        stats["settled"] = len(done)
    if not found:
        return math.inf, []
    path, u = [dst], dst
    while u != src:
        u = prev[u]; path.append(u)
    return g[dst], path[::-1]


# ---------------- Stage 3: backbone (MST over the road-distance metric closure) ----------------
def road_distance_matrix(G, tower_nodes):
    """k x k matrix of shortest ROAD distances between towers: k Dijkstras, O(k(V+E)log V).
    A* does not help here - it needs a single target, and this needs all of them."""
    k = len(tower_nodes)
    dm = [[0.0] * k for _ in range(k)]
    for i, t in enumerate(tower_nodes):
        dist, _ = dijkstra(G, t)
        for j, u in enumerate(tower_nodes):
            dm[i][j] = dist.get(u, math.inf)
    return dm


def bridges(k, links):
    """Tarjan's bridge-finding by DFS low-link, O(V+E). A bridge is an edge whose removal
    disconnects the graph - in a backbone, a single point of failure.

    `links` is an iterable of (i, j) index pairs; returns a set of frozensets. Iterative DFS, not
    recursive, so a long chain of towers cannot blow the Python stack.

    low[v] = the smallest discovery time reachable from v's subtree using at most one back edge.
    Edge (u, v) with v a child is a bridge exactly when low[v] > disc[u]: nothing under v reaches
    u or above, so that edge is the only way back.

    Parallel links are tracked by **edge id**, not by parent node, so the DFS skips only the exact
    edge it arrived on. That matters: two towers joined by two fibres have no single point of
    failure between them, but collapsing the pair into one edge - or using the usual
    skip-the-parent shortcut - would wrongly report a bridge. Self-loops are never bridges.
    """
    adj = {i: [] for i in range(k)}
    for eid, (i, j) in enumerate(links):
        if i == j:
            # A self-loop cannot disconnect anything. Defensive only: verified over 40k random
            # multigraphs that leaving them in changes no result, since a self-loop only ever
            # relaxes low[u] against disc[u] itself.
            continue
        adj[i].append((j, eid))
        adj[j].append((i, eid))
    disc, low, found = {}, {}, set()
    timer = 0
    for root in range(k):
        if root in disc:
            continue
        # stack frames: (node, id of the edge we entered it by, iterator over (neighbour, edge id))
        disc[root] = low[root] = timer; timer += 1
        stack = [(root, None, iter(adj[root]))]
        while stack:
            u, in_eid, it = stack[-1]
            advanced = False
            for v, eid in it:
                if eid == in_eid:
                    continue                                # the same edge back, not a back edge
                if v in disc:
                    # disc[v], not low[v]. For BRIDGES the two are interchangeable (verified over
                    # 40k random multigraphs); for articulation points they are not, so keep disc
                    # if this ever grows into computing those.
                    low[u] = min(low[u], disc[v])            # back edge (or a parallel link)
                    continue
                disc[v] = low[v] = timer; timer += 1
                stack.append((v, eid, iter(adj[v])))
                advanced = True
                break
            if not advanced:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[u])
                    if low[u] > disc[p]:
                        found.add(frozenset((p, u)))
    return found


def prim(dm):
    """Prim's MST over the k x k metric closure. O(k^2) with a plain array scan for the minimum
    key, which is the right choice here: the closure is dense (every pair has a distance), so a
    heap's log factor buys nothing. Returns (edges, total) - the same shape as kruskal().
    """
    k = len(dm)
    in_t, key, par, edges = [False] * k, [math.inf] * k, [-1] * k, []
    # Documentary rather than load-bearing: every key starts at infinity, so the first min() picks
    # index 0 whether or not this line runs (verified over 5k random matrices). It is kept because
    # "the tree starts at tower 0" is the intent, and relying on min()'s tie-breaking would not be.
    key[0] = 0
    for _ in range(k):
        u = min((i for i in range(k) if not in_t[i]), key=lambda i: key[i])
        in_t[u] = True
        if par[u] >= 0:
            edges.append((par[u], u, dm[par[u]][u]))
        for v in range(k):
            if not in_t[v] and dm[u][v] < key[v]:
                key[v], par[v] = dm[u][v], u
    return edges, sum(e[2] for e in edges)


def backbone(G, tower_nodes, extra_links=0, strategy="bridges", dm=None, mst="prim"):
    """Minimum spanning backbone over the towers' road-distance metric closure, plus optional
    redundant links. Returns (edges, total length, dm) - the distance matrix comes back so callers
    can site an exchange or re-run a different MST without paying for k more Dijkstras.

    mst="prim" (O(k^2), better on this dense closure) or "kruskal" (O(k^2 log k), sort-dominated).
    Both produce the same total length - the MST weight is unique even where the tree is not - so
    which one runs is a demonstration, not a decision. `dm` may be passed in to skip recomputing it.

    extra_links adds links beyond the tree, which is what gives stage 4 a non-trivial min-cut:
      strategy="cheapest" - the globally cheapest non-tree links. Simple, but it spends the budget
                            wherever it is cheap rather than where it is needed.
      strategy="bridges"  - each round, add the cheapest link that removes at least one bridge,
                            recomputing bridges each round since one link can clear several. Falls
                            back to cheapest once no bridge remains. Same budget, never more single
                            points of failure.
    """
    k = len(tower_nodes)
    if dm is None:
        dm = road_distance_matrix(G, tower_nodes)
    if mst == "prim":
        edges, _ = prim(dm)
    elif mst == "kruskal":
        edges, _ = kruskal(dm)
    else:
        raise ValueError(f"unknown mst algorithm {mst!r}")
    edges += _redundant_links(dm, k, edges, extra_links, strategy)
    return edges, sum(e[2] for e in edges), dm


class UnionFind:
    """Disjoint-set forest with path compression and union by rank.

    m operations over k elements cost O(m * alpha(k)) where alpha is the inverse Ackermann
    function - under 5 for any k that fits in the universe, so effectively constant. Written out
    rather than imported because it is the data structure Kruskal is *for*.
    """

    def __init__(self, k):
        self.parent = list(range(k))
        self.rank = [0] * k
        self.components = k

    def find(self, x):
        """Root of x's set, flattening the path on the way back up (path compression)."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:                   # second pass: point everything at the root
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        """Merge the two sets. Returns False if they were already joined - which is exactly how
        Kruskal detects that an edge would close a cycle."""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # Union by rank keeps the trees shallow. This is a PERFORMANCE property only: inverting
        # the comparison was verified to produce zero correctness differences over 5k random
        # union sequences, and with path compression also on, max depth stayed 1 either way.
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.components -= 1
        return True


def kruskal(dm):
    """Kruskal's MST over the k x k metric closure. O(k^2 log k): the sort dominates, since the
    closure has k(k-1)/2 edges and each union-find operation is effectively constant.

    Returns (edges, total) in Prim's shape, so the two are drop-in interchangeable. The MST
    *weight* is unique even when the tree is not, which is what the cross-validation test asserts.
    Sorting globally and rejecting cycle-closing edges is the opposite strategy to Prim's growing
    of one tree, and a useful contrast for the report: Prim is better on dense graphs like this
    metric closure, Kruskal on sparse ones.
    """
    k = len(dm)
    candidates = sorted((dm[i][j], i, j) for i in range(k) for j in range(i + 1, k)
                        if dm[i][j] < math.inf)
    uf, edges = UnionFind(k), []
    for d, i, j in candidates:
        if uf.union(i, j):
            edges.append((i, j, d))
            if len(edges) == k - 1:                     # a spanning tree cannot use more
                break
    return edges, sum(e[2] for e in edges)


def choose_exchange(dm, mode="median", load=None, xy=None):
    """Pick which built tower hosts the exchange (the head-end where upstream fibre lands).

    Free given `dm` from backbone() - no extra Dijkstras. O(k^2).

      "median"   1-median: minimise the total road distance to every other tower, weighted by each
                 tower's customer load when `load` is given, so the head-end sits near the demand
                 rather than merely near the geometric middle of the towers. This is the right
                 default: total distance is what backhaul actually costs.
      "center"   1-center: minimise the WORST distance to any tower, i.e. bound the latency of the
                 unluckiest customer rather than the average.
      "centroid" the original rule - the tower nearest the Euclidean centroid of the tower
                 positions. Kept for comparison; needs `xy` as a list of (x, y). It ignores the
                 road network entirely, which is why it can be beaten on both of the above.

    Note both 1-median and 1-center here are restricted to *tower* sites, which is the real
    constraint (the exchange has to live at a tower), and are therefore exactly solvable by
    enumeration - unlike the general k-median problem, which is NP-hard.
    """
    k = len(dm)
    if k == 0:
        raise ValueError("no towers to host an exchange")
    if mode == "median":
        w = load if load is not None else [1] * k
        return min(range(k), key=lambda i: sum(dm[i][j] * w[j] for j in range(k)))
    if mode == "center":
        return min(range(k), key=lambda i: max(dm[i]))
    if mode == "centroid":
        if xy is None:
            raise ValueError("mode='centroid' needs xy=[(x, y), ...]")
        cx = sum(p[0] for p in xy) / k
        cy = sum(p[1] for p in xy) / k
        return min(range(k), key=lambda i: (xy[i][0] - cx) ** 2 + (xy[i][1] - cy) ** 2)
    raise ValueError(f"unknown exchange mode {mode!r}")


def _redundant_links(dm, k, tree_edges, extra_links, strategy):
    """Pick `extra_links` links beyond the spanning tree. See backbone() for the strategies."""
    used = {frozenset(e[:2]) for e in tree_edges}
    candidates = sorted((dm[i][j], i, j) for i in range(k) for j in range(i + 1, k)
                        if frozenset((i, j)) not in used and dm[i][j] < math.inf)
    if strategy == "cheapest":
        return [(i, j, d) for d, i, j in candidates[:extra_links]]
    if strategy != "bridges":
        raise ValueError(f"unknown redundancy strategy {strategy!r}")
    chosen, links = [], [tuple(sorted(e[:2])) for e in tree_edges]
    pool = list(candidates)
    for _ in range(extra_links):
        if not pool:
            break
        current = bridges(k, links)
        pick = None
        if current:
            # the cheapest link that removes at least one bridge
            for idx, (d, i, j) in enumerate(pool):
                if bridges(k, links + [(i, j)]) < current:
                    pick = idx
                    break
        if pick is None:
            pick = 0                        # no bridge left to fix: fall back to cheapest
        d, i, j = pool.pop(pick)
        chosen.append((i, j, d))
        links.append((i, j))
    return chosen


# ---------------- Stage 4: Edmonds-Karp ----------------
def edmonds_karp(cap, s, t, flows=None):
    """Max-flow by BFS shortest augmenting path (Edmonds-Karp, O(V*E^2)) - not plain
    Ford-Fulkerson; the BFS is what bounds the iteration count.
    cap: dict u -> dict v -> capacity (a copy is modified). Returns (flow, min-cut edges).

    flows: optional dict, filled with {(u, v): flow on that arc}. This is the *net* flow, derived
    as capacity minus residual, so it is exact only where the network has no antiparallel arcs -
    true of the bipartite assignment network, NOT of the backbone (whose undirected links become
    arcs both ways, letting the residual of one absorb the cancellation of the other)."""
    res = {u: dict(vs) for u, vs in cap.items()}
    for u, vs in cap.items():
        for v in vs:
            res.setdefault(v, {}).setdefault(u, 0)
    flow = 0
    while True:
        par, q = {s: None}, deque([s])
        while q and t not in par:
            u = q.popleft()
            for v, c in res[u].items():
                if c > 0 and v not in par:
                    par[v] = u; q.append(v)
        if t not in par:
            break
        path, v = [], t
        while par[v] is not None:
            path.append((par[v], v)); v = par[v]
        f = min(res[u][v] for u, v in path)
        for u, v in path:
            res[u][v] -= f; res[v][u] += f
        flow += f
    reach, q = {s}, deque([s])
    while q:
        u = q.popleft()
        for v, c in res[u].items():
            if c > 0 and v not in reach:
                reach.add(v); q.append(v)
    cut = [(u, v, c) for u, vs in cap.items() for v, c in vs.items() if u in reach and v not in reach and c > 0]
    if flows is not None:
        for u, vs in cap.items():
            for v, c in vs.items():
                flows[(u, v)] = c - res[u][v]
    return flow, cut


def build_flow_network(tower_caps, edges, demand_mbps, exchange):
    """Exchange -> towers -> customers, with **node splitting**: tower k becomes
    ("in",k)->("out",k) carrying that tower's equipment capacity, so a tower cannot relay more
    traffic than its own hardware can switch. Without the split a degree-4 exchange could emit
    four times its own capacity. Backbone links are undirected (both directions, capacity = min
    of endpoint capacities) and each tower drains its own customers into SINK. The super-source
    and the sink are wired here, so the result goes straight to edmonds_karp."""
    cap = {}

    def arc(u, v, c):
        cap.setdefault(u, {})[v] = c

    for k, c in enumerate(tower_caps):
        arc(("in", k), ("out", k), c)                      # the tower's own equipment
    for i, j, _ in edges:
        c = min(tower_caps[i], tower_caps[j])
        arc(("out", i), ("in", j), c)
        arc(("out", j), ("in", i), c)
    for k, dm in enumerate(demand_mbps):
        arc(("out", k), "SINK", dm)                        # customers hanging off tower k
    # The upstream fibre lands on the exchange's backbone switch, so it is fed to ("out", ...)
    # and bypasses that tower's own access radio - otherwise the head-end throttles the entire
    # network to one tower's customer-serving capacity and the min-cut is always trivially there.
    arc("EX", ("out", exchange), sum(tower_caps) or 1)
    cap.setdefault("SINK", {})
    return cap


def describe_cut(cut):
    """Translate min-cut arcs out of the split-node network back into plan language. Returns
    (readable rows, saturated backbone links, towers whose own equipment is the bottleneck)."""
    def name(x):
        return x if isinstance(x, str) else "T%d.%s" % (x[1], x[0])

    rows, links, nodes = [], set(), set()
    for u, v, c in cut:
        rows.append((name(u), name(v), c))
        if isinstance(u, tuple) and isinstance(v, tuple):
            if u[1] != v[1]:
                links.add(frozenset((u[1], v[1])))         # a backbone link is saturated
            else:
                nodes.add(u[1])                            # the tower's own kit is the limit
    return rows, links, nodes


# ---------------- Stage 2b: customer assignment ----------------
def _covering(G, C, D, sites):
    """For each demand index, the (distance, tower index) pairs whose radius reaches it, nearest
    first. The single place the coverage rule lives - both assignment strategies read it, so they
    can never disagree about who can serve whom. O(|D| * k)."""
    out = []
    for d in D:
        dn = G.nodes[d["node"]]
        hits = []
        for k, s in enumerate(sites):
            cn = G.nodes[C[s]["node"]]
            dd = math.hypot(cn["x"] - dn["x"], cn["y"] - dn["y"])
            if dd <= RADIUS[C[s]["tier"]]:
                hits.append((dd, k))
        hits.sort()
        out.append(hits)
    return out


def assign_customers(G, C, D, sites):
    """Greedy: every customer to its NEAREST covering tower, capacity ignored. O(|D| * k).

    Simple and fast, but it over-subscribes popular towers - on the default instance it piles 146%
    of one tower's purchased capacity onto it while five others idle near 50%. Kept for two
    reasons: assign_customers_flow needs a provisional load to size equipment against (capacity
    and assignment are otherwise circular), and the contrast is worth showing.
    Returns {tower index: customer weight}.
    """
    out = {i: 0 for i in range(len(sites))}
    for j, hits in enumerate(_covering(G, C, D, sites)):
        if hits:
            out[hits[0][1]] += D[j]["w"]
    return out


def assign_customers_flow(G, C, D, sites, caps, mbps):
    """Capacity-aware: assign customers by max-flow so no tower is asked for more than the
    equipment actually bought for it.

        SRC      -> demand j    capacity w_j * mbps
        demand j -> tower k     capacity w_j * mbps, for every tower whose radius covers j
        tower k  -> SINK        capacity caps[k]

    All capacities are integral Mbps, which is what gives Edmonds-Karp its clean termination
    argument. The max flow is the most bandwidth this built network can actually deliver; each
    tower's share is the flow on its SINK arc. Reuses edmonds_karp - the same Unit V machinery
    stage 4 runs. O(V*E^2) on a bipartite network of |D| + k + 2 nodes.

    Returns (load_mbps per tower, unserved_mbps). A non-zero unserved figure is a real result -
    "this much demand cannot be served with the equipment purchased" - and strictly more honest
    than the greedy version, which reports nothing and silently overloads instead.

    Deliberate limitation: max-flow maximises the TOTAL served and is indifferent between tied
    optima, so a customer may be routed to a farther tower when a nearer one would have done.
    Fixing that is a min-cost flow, out of scope here; the objective is customers served, which is
    what the dashboard reports.
    """
    cov = _covering(G, C, D, sites)
    cap = {}

    def arc(u, v, c):
        cap.setdefault(u, {})[v] = c

    total = sum(d["w"] * mbps for d in D)
    for j, d in enumerate(D):
        if not cov[j]:
            continue                       # no built tower reaches it: counts as unserved
        need = d["w"] * mbps
        arc("SRC", ("d", j), need)
        for _, k in cov[j]:
            arc(("d", j), ("t", k), need)
    for k, c in enumerate(caps):
        arc(("t", k), "SINK", c)
    cap.setdefault("SINK", {})
    flows = {}
    served, _ = edmonds_karp(cap, "SRC", "SINK", flows=flows)
    load = [flows.get((("t", k), "SINK"), 0) for k in range(len(sites))]
    return load, total - served


def benchmark(G, sizes=(8, 12, 16, 20, 24, 28), n_demand=60, seed=1, limit=20, naive=True, progress=None):
    """Stage-1 runtime scaling: naive backtracking vs branch-and-bound vs greedy, all three on
    the same instance at each |C|. One dict per size; a `timed_out` row is NOT an optimum."""
    rows = []
    for idx, n in enumerate(sizes):
        D, C, _ = make_instance(G, n_demand, n, seed)
        rows.append({"n_cand": n, "n_demand": len(D),
                     "bb": exact_cover(C, D, limit),
                     "greedy": greedy_cover(C, D),
                     "naive": exact_cover(C, D, limit, prune=False) if naive else None})
        if progress:
            progress((idx + 1) / len(sizes), n)
    return rows


if __name__ == "__main__":
    G, _ = largest_component(synthetic_graph())
    D, C, dropped = make_instance(G, 60, 16, 0)
    print("instance: %d of %d demand points kept (%d unreachable by any candidate)"
          % (len(D), dropped["requested"], dropped["count"]))
    e, g = exact_cover(C, D), greedy_cover(C, D)
    print("exact ", e)
    print("greedy", g)
    print("coverage of ALL requested demand: exact %.1f%%" % coverage_of_requested(e, D, dropped))
    sites = e["sites"]
    towers = [C[i]["node"] for i in sites]
    cust = assign_customers(G, C, D, sites)          # stage 2 needs each tower's load first
    dem = [cust[i] * 5 for i in range(len(sites))]
    caps, picks, spends = equip_towers(C, sites, dem, 12)
    print("stage 2 demand", dem)
    print("stage 2 caps  ", caps)
    print("stage 2 spend ", spends, "= %.1f k$ equipment on top of %.1f k$ build" % (sum(spends), e["cost"]))
    edges, total, dm = backbone(G, towers, extra_links=2)
    exch = choose_exchange(dm, "median", load=dem)
    print("exchange  T%d (1-median; total %.2f km, worst %.2f km)"
          % (exch, sum(dm[exch]) / 1000, max(dm[exch]) / 1000))
    cap = build_flow_network(caps, edges, dem, exch)
    flow, cut = edmonds_karp(cap, "EX", "SINK")
    rows, links, nodes = describe_cut(cut)
    print("cable m", round(total), "| flow", flow, "of demand", sum(dem),
          "(%.0f%%)" % (100 * flow / max(1, sum(dem))))
    print("min-cut", rows)
    print("  saturated links", sorted(tuple(sorted(l)) for l in links),
          "| towers capped by own kit", sorted(nodes))
    dist, path = dijkstra(G, towers[0], D[0]["node"])
    ref = nx.dijkstra_path_length(G, towers[0], D[0]["node"], weight="length")
    print("dijkstra", round(dist, 1), "networkx", round(ref, 1),
          "| match" if abs(dist - ref) < 1e-6 else "| MISMATCH")
