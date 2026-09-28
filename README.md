# ISP Network Planner

Plans a mobile/internet network for a city: which towers to build, what equipment to put on
them, how to wire them together, and where the bandwidth bottleneck ends up. Every algorithm is
hand-written; NetworkX is only a graph container and a correctness reference.

## Run in VS Code

1. Open this folder in VS Code (File > Open Folder).
2. Terminal > New Terminal, then:
   ```
   python -m venv venv
   venv\Scripts\activate        (Windows)   |   source venv/bin/activate   (Mac/Linux)
   pip install -r requirements.txt
   ```
3. Quick console test: `python planner.py`
4. Full app: `streamlit run app.py`

Choose "Synthetic grid (offline)" in the sidebar first; use "OpenStreetMap place" when online.

## Files

| File | Owner | Contents |
|---|---|---|
| `interface.py` | both | The locked A/B data contract (plan Section 3) + all tunable constants |
| `planner.py` | A (stages 0-2), B (stages 3-5) | Every algorithm |
| `app.py` | B | Stage 6: Streamlit + Folium dashboard |

`interface.py` is the artifact to check at every sync. Read it before changing any stage.

## Stages, algorithms, complexity

| Stage | What it does | Algorithm | Complexity | Unit |
|---|---|---|---|---|
| 0 | Load road network, demand, candidate sites | Graph build + BFS connectivity (`components`) | O(V+E) | I |
| 1 | Cheapest tower set covering every customer | Greedy weighted set cover (`greedy_cover`) | O(\|C\|² · \|D\|/64), (ln n + 1)-approx | IV + VI |
| 1 | ...and the true optimum | Backtracking + branch and bound (`exact_cover`) | O(2^\|C\|) worst case | IV + VI |
| 2 | Equipment per tower under budget | 0/1 knapsack DP (`knapsack`, `equip_towers`) | O(n · B) per tower | III |
| 3 | Wire towers with least cable | Prim MST over the road-distance metric closure (`backbone`) | k Dijkstras + O(k²) | II / IV |
| 4 | Push bandwidth from exchange, find bottleneck | Edmonds–Karp max-flow / min-cut (`edmonds_karp`) | O(V · E²) | V |
| 5 | Route signal tower → customer | Binary-heap Dijkstra (`dijkstra`) | O((V+E) log V) | II |
| 6 | Dashboard | — | — | Result |

## Stage 1 is the headline: measured results

`|D|=60`, seed 0, synthetic grid, 5 s limit. Naive backtracking is the *identical recursion* with
every bound removed, so the node counts are directly comparable.

| \|C\| | naive nodes | B&B nodes | nodes saved | exact k$ | greedy k$ | greedy gap |
|---|---|---|---|---|---|---|
| 8 | 511 | 22 | 23× | 67.26 | 67.26 | 0.0% |
| 12 | 8,191 | 38 | 216× | 76.61 | 84.72 | 10.6% |
| 16 | 127,935 | 85 | 1,505× | 77.40 | 91.16 | 17.8% |
| 20 | 1,946,815 | 456 | 4,269× | 80.93 | 110.24 | 36.2% |
| 24 | 10,047,287 **(timed out)** | 676 | — | 77.31 | 85.41 | 10.5% |
| 28 | 9,676,189 **(timed out)** | 572 | — | 74.14 | 96.24 | 29.8% |

Two things for the report and the viva:

- **Naive backtracking hits the 2ⁿ wall at |C| = 24**, which is the plan's "stalls past ~15–20
  candidate sites" claim. Branch and bound stays under 5 ms up to |C| = 28 and under 0.12 s at
  |C| = 40 (8,142 nodes, measured). Pruning is not optional — it is the difference between a live
  demo and a hang.
- **The pruning is lossless.** Verified over 18 instances (6 seeds × |C| = 10/14/16): `prune=False`
  finds exactly the same optimal cost as `prune=True`, using 2,000–131,000 nodes instead of
  22–242. The bounds cut only branches that provably cannot improve the incumbent.

Greedy is strictly costlier than the optimum on 8 of 10 seeds, by up to 62%.

## Deliberate deviations from the plan of action

Record these in the report so the writeup matches the code.

1. **Stage 4 is Edmonds–Karp, not plain Ford–Fulkerson.** The augmenting path is found by BFS
   (shortest path), which is what bounds the iteration count at O(V·E²); plain Ford–Fulkerson has
   no such bound. Do not claim Ford–Fulkerson in the report.
2. **Stage 4 splits every tower node.** Tower `k` becomes `("in",k) → ("out",k)` with capacity
   equal to its equipment capacity, so a tower cannot relay more traffic than its own hardware
   switches. Without the split a degree-4 exchange could emit four times its own capacity. The
   upstream fibre is fed to `("out", exchange)` because it lands on the backbone switch rather
   than passing through that tower's access radio.
3. **Stage 3 implements Prim only**; the plan said "Prim's / Kruskal's".
4. **Stage 5 implements Dijkstra only**; the plan said "Dijkstra / A*". A* with a Euclidean
   heuristic on the projected coordinates would be admissible (road length ≥ straight-line
   distance) and is a cheap addition if a second Stage 5 comparison is wanted.
5. **Stage 0 uses BFS only** for the connectivity check; the plan said "DFS/BFS".
6. **The shared data keys differ from the plan's Section 3 wording** (`node` not `position`, `cost`
   not `build_cost`, `w` not `weight`, no explicit `eid`). The mapping is tabulated at the top of
   `interface.py`.

## Reading the dashboard honestly

- **"Customers covered"** is the fraction of *all requested* demand weight. It is not the same as
  the solvers' own `coverage_pct`, which is 100% of the demand that survived into the set-cover
  instance. `make_instance` must drop demand that no candidate site can reach, and reports how
  much in `dropped`; the dashboard warns when it is non-zero.
- **A timed-out solver row is not an optimum.** The benchmark chart rings those points in red.
- **"Project cost"** is tower build cost *plus* stage-2 equipment spend. Stage 1's own `cost` field
  is build cost only.

## Calibration

`interface.py` holds `RADIUS`, `BASE_COST`, `OPTIONS`, `TIER_BONUS` and `HEADROOM`. They are tuned
together — changing one in isolation will make a stage look broken:

- Radii too small → most demand is unreachable and most candidates become *forced*, which leaves
  stage 1 with no combinatorial choice and makes exact and greedy return identical costs.
- Radii too large → a handful of towers cover everything, and stage 1 goes trivial again.
- `HEADROOM` below ~1.5 → every tower is sized to exactly its own customers, leaving no transit
  capacity, and the backbone starves (stage 4 served 19% of demand at headroom 1.0).

## Still to do

- **No version control.** The plan names format drift between A and B as risk #1 and mandates
  syncs every 2–3 days. `git init` and push before the next sync.
- **No test suite.** The hand-written algorithms should be checked against NetworkX references
  (`nx.minimum_spanning_tree`, `nx.maximum_flow`, `nx.dijkstra_path_length`) and the exact solver
  against brute force on small instances. Right now only Dijkstra is verified, at the bottom of
  `planner.py`.
- Milestone 7 wants saved benchmark artifacts (CSV/PNG) for the report; `benchmark()` returns the
  rows but nothing writes them to disk.
