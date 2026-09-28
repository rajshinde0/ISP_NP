"""ISP Network Planner - stages 0-5 (all algorithms hand-written; networkx is only a graph container)."""
import heapq, math, random, time
from collections import deque
import networkx as nx

RADIUS = {"mid": 400.0, "high": 900.0}      # metres
BASE_COST = {"mid": 10.0, "high": 28.0}     # k$
OPTIONS = [(2, 10), (4, 25), (7, 50), (10, 80)]  # (cost k$, capacity Mbps)
LAT0, LON0 = 18.5204, 73.8567


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
    keep = [j for j in range(len(D)) if covered >> j & 1]  # drop demand no candidate can reach
    if len(keep) < len(D):
        D = [D[j] for j in keep]
        for c in C:
            c["mask"] = sum(1 << k for k, j in enumerate(keep) if c["mask"] >> j & 1)
    return D, C


def wmask(m, w):
    s = 0
    while m:
        b = m & -m
        s += w[b.bit_length() - 1]
        m ^= b
    return s


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
    """Cost-weighted greedy: max (newly covered weight / cost). Approximation ratio H(n) <= ln(n)+1."""
    t0 = time.perf_counter()
    w = [d["w"] for d in D]
    unc, chosen = (1 << len(D)) - 1, []
    while unc:
        best, bi = 0, None
        for i, c in enumerate(C):
            if i in chosen:
                continue
            g = wmask(c["mask"] & unc, w) / c["cost"]
            if g > best:
                best, bi = g, i
        if bi is None:
            break
        chosen.append(bi)
        unc &= ~C[bi]["mask"]
    return _result("greedy", C, D, chosen, t0)


def exact_cover(C, D, time_limit=30.0):
    """Branch-and-bound backtracking. Pruning: infeasible, cost >= incumbent, fractional lower bound."""
    t0 = time.perf_counter()
    w = [d["w"] for d in D]
    full, n = (1 << len(D)) - 1, len(C)
    order = sorted(range(n), key=lambda i: C[i]["cost"] / max(1, wmask(C[i]["mask"], w)))
    masks, costs = [C[i]["mask"] for i in order], [C[i]["cost"] for i in order]
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] | masks[i]
    g = greedy_cover(C, D)  # incumbent -> strong initial bound
    best = [g["cost"] + 1e-9, list(g["sites"])]
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
        if i == n or cost >= best[0] or (cov | suffix[i]) != full:
            return
        unc = full & ~cov
        ratios = [costs[j] / wmask(masks[j] & unc, w) for j in range(i, n) if masks[j] & unc]
        if cost + min(ratios) * wmask(unc, w) >= best[0]:
            return
        if masks[i] & unc:
            rec(i + 1, cov | masks[i], cost + costs[i], chosen + [i])
        rec(i + 1, cov, cost, chosen)

    rec(0, 0, 0.0, [])
    return _result("exact (B&B)", C, D, best[1], t0, {"nodes": state["nodes"], "timed_out": state["timeout"]})


# ---------------- Stage 2: knapsack ----------------
def knapsack(options=OPTIONS, budget=12):
    """0/1 knapsack, dp[i][b] = max capacity using first i options within budget b. O(n*B)."""
    n = len(options)
    dp = [[0] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        c, cap = options[i - 1]
        for b in range(budget + 1):
            dp[i][b] = dp[i - 1][b]
            if c <= b and dp[i - 1][b - c] + cap > dp[i][b]:
                dp[i][b] = dp[i - 1][b - c] + cap
    b, picked = budget, []
    for i in range(n, 0, -1):
        if dp[i][b] != dp[i - 1][b]:
            picked.append(options[i - 1]); b -= options[i - 1][0]
    return dp[n][budget], picked[::-1]


# ---------------- Stage 5 (used by stage 3): Dijkstra ----------------
def dijkstra(G, src, dst=None):
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
    return edges, sum(e[2] for e in edges if True)


# ---------------- Stage 4: Edmonds-Karp ----------------
def edmonds_karp(cap, s, t):
    """cap: dict u -> dict v -> capacity (modified copy used). Returns (flow, min-cut edges, flow per edge)."""
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
    """Undirected backbone links (cap = min endpoint capacity) + per-tower sink edges."""
    cap = {}
    for i, j, _ in edges:
        c = min(tower_caps[i], tower_caps[j])
        cap.setdefault(i, {})[j] = c
        cap.setdefault(j, {})[i] = c
    for i, dm in enumerate(demand_mbps):
        cap.setdefault(i, {})["SINK"] = min(dm, tower_caps[i])
    return cap


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


def benchmark(G, sizes=(8, 10, 12, 14, 16, 18), n_demand=40, seed=1, limit=20):
    rows = []
    for n in sizes:
        D, C = make_instance(G, n_demand, n, seed)
        rows.append((n, exact_cover(C, D, limit), greedy_cover(C, D)))
    return rows


if __name__ == "__main__":
    G, _ = largest_component(synthetic_graph())
    D, C = make_instance(G, 60, 16, 0)
    e, g = exact_cover(C, D), greedy_cover(C, D)
    print("exact ", e); print("greedy", g)
    sites = e["sites"]
    towers = [C[i]["node"] for i in sites]
    caps = [knapsack(budget=12)[0]] * len(sites)
    edges, total = backbone(G, towers, extra_links=2)
    cust = assign_customers(G, C, D, sites)
    dem = [cust[i] * 5 for i in range(len(sites))]
    cap = build_flow_network(caps, edges, dem, 0)
    cap.setdefault("SINK", {})
    flow, cut = edmonds_karp(cap, 0, "SINK")
    print("cable m", round(total), "| flow", flow, "of demand", sum(dem), "| min-cut", cut)
    dist, path = dijkstra(G, towers[0], D[0]["node"])
    ref = nx.dijkstra_path_length(G, towers[0], D[0]["node"], weight="length")
    print("dijkstra", round(dist, 1), "networkx", round(ref, 1))
