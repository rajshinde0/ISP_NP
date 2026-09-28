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


def osm_graph(place):
    import osmnx as ox
    G0 = ox.graph_from_place(place, network_type="drive")
    Gp = ox.project_graph(G0)  # metres, not degrees
    H = nx.Graph()
    for n, d in Gp.nodes(data=True):
        H.add_node(n, x=d["x"], y=d["y"], lat=G0.nodes[n]["y"], lon=G0.nodes[n]["x"])
    for u, v, d in Gp.edges(data=True):
        l = d["length"]
        if H.has_edge(u, v):
            l = min(l, H[u][v]["length"])
        H.add_edge(u, v, length=l)
    return H


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


# ---------------- Stage 5 (used by stage 3): Dijkstra ----------------
def dijkstra(G, src, dst=None):
    """Binary-heap Dijkstra on edge "length". O((V+E) log V).
    dst=None -> (dist, prev) over the whole reachable set; else (distance, path)."""
    dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        if u == dst:
            break
        for v, e in G[u].items():
            nd = d + e["length"]
            if nd < dist.get(v, math.inf):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if dst is None:
        return dist, prev
    if dst not in dist:
        return math.inf, []
    path, u = [dst], dst
    while u != src:
        u = prev[u]; path.append(u)
    return dist[dst], path[::-1]


# ---------------- Stage 3: backbone (Prim on road-distance metric closure) ----------------
def backbone(G, tower_nodes, extra_links=0):
    """Prim MST over the towers' road-distance metric closure: k Dijkstras to build the k x k
    matrix, then O(k^2) for Prim. extra_links adds the cheapest non-tree links back for
    redundancy, which is what gives stage 4 a non-trivial min-cut to find."""
    k = len(tower_nodes)
    dm = [[0.0] * k for _ in range(k)]
    for i, t in enumerate(tower_nodes):
        dist, _ = dijkstra(G, t)
        for j, u in enumerate(tower_nodes):
            dm[i][j] = dist.get(u, math.inf)
    in_t, key, par, edges = [False] * k, [math.inf] * k, [-1] * k, []
    key[0] = 0
    for _ in range(k):
        u = min((i for i in range(k) if not in_t[i]), key=lambda i: key[i])
        in_t[u] = True
        if par[u] >= 0:
            edges.append((par[u], u, dm[par[u]][u]))
        for v in range(k):
            if not in_t[v] and dm[u][v] < key[v]:
                key[v], par[v] = dm[u][v], u
    used = {frozenset(e[:2]) for e in edges}
    extras = sorted(((dm[i][j], i, j) for i in range(k) for j in range(i + 1, k) if frozenset((i, j)) not in used and dm[i][j] < math.inf))
    for d, i, j in extras[:extra_links]:  # redundancy -> cycles -> non-trivial min-cut
        edges.append((i, j, d))
    return edges, sum(e[2] for e in edges)


# ---------------- Stage 4: Edmonds-Karp ----------------
def edmonds_karp(cap, s, t):
    """Max-flow by BFS shortest augmenting path (Edmonds-Karp, O(V*E^2)) - not plain
    Ford-Fulkerson; the BFS is what bounds the iteration count.
    cap: dict u -> dict v -> capacity (a copy is modified). Returns (flow, min-cut edges)."""
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


# ---------------- driver ----------------
def assign_customers(G, C, D, sites):
    out = {i: 0 for i in range(len(sites))}
    for d in D:
        dn = G.nodes[d["node"]]
        best = None
        for k, s in enumerate(sites):
            cn = G.nodes[C[s]["node"]]
            dd = math.hypot(cn["x"] - dn["x"], cn["y"] - dn["y"])
            if dd <= RADIUS[C[s]["tier"]] and (best is None or dd < best[0]):
                best = (dd, k)
        if best:
            out[best[1]] += d["w"]
    return out


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
    edges, total = backbone(G, towers, extra_links=2)
    cap = build_flow_network(caps, edges, dem, 0)
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
