# ISP Network Planner

Plans a mobile/internet network for a city: which towers to build, what equipment to put on
them, how to wire them together, and where the bandwidth bottleneck ends up. Every algorithm is
hand-written; NetworkX is only a graph container and a correctness oracle for the tests.

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
5. Tests: `pytest -q`
6. Report figures: `python bench.py`

Choose "Synthetic grid (offline)" in the sidebar first; use "OpenStreetMap place" when online.

## Files

| File | Owner | Contents |
|---|---|---|
| `interface.py` | both | The locked A/B data contract (plan Section 3) + all tunable constants |
| `planner.py` | A (stages 0-2), B (stages 3-5) | Every algorithm |
| `app.py` | B | Stage 6: Streamlit + Folium dashboard |
| `bench.py` | both | Stage-1 benchmark CLI; writes the report's CSV/PNG/Markdown |
| `tests/` | both | Correctness tests against NetworkX and brute force |
| `docs/np-hardness.md` | both | Milestone 7: the NP-hardness writeup |

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

Produced by `python bench.py` (`bench/stage1_scaling.csv`). 60 demand points requested, seed 0,
synthetic grid, 20 s limit. Naive backtracking is the *identical recursion* with every bound
removed, so node counts are directly comparable.

| \|C\| | \|D\| | naive nodes | naive s | B&B nodes | B&B s | nodes saved | exact k$ | greedy k$ | greedy gap |
|---|---|---|---|---|---|---|---|---|---|
| 8 | 44 | 511 | 0.0002 | 22 | 0.0001 | 23× | 67.26 | 67.26 | 0.0% |
| 12 | 60 | 8,191 | 0.004 | 38 | 0.0002 | 216× | 76.61 | 84.72 | 10.6% |
| 16 | 57 | 127,935 | 0.052 | 85 | 0.0005 | 1,505× | 77.40 | 91.16 | 17.8% |
| 20 | 60 | 1,946,815 | 0.850 | 456 | 0.0030 | 4,269× | 80.93 | 110.24 | 36.2% |
| 24 | 60 | 30,799,999 | 13.13 | 676 | 0.0040 | 45,562× | 77.31 | 85.41 | 10.5% |
| 28 | 60 | 46,284,591 **(timed out)** | 20.0 | 572 | 0.0053 | — | 74.14 | 96.24 | 29.8% |

Four things for the report and the viva:

- **Naive backtracking grows about 16× per four extra candidates** — that is 2⁴, the textbook 2ⁿ
  curve — and hits the 20 s wall at |C| = 28. This is the plan's "stalls past ~15–20 candidate
  sites" claim. Where exactly the wall falls depends on the time limit: at a 5 s limit it already
  fails at |C| = 24.
- **Branch and bound never stalls.** Under 6 ms through |C| = 28, and 0.12 s at |C| = 40. Pruning
  is not optional — it is the difference between a live demo and a hang.
- **A timed-out search is not an optimum, and this table proves it.** At |C| = 28 naive's truncated
  incumbent is 86.32 k$ while B&B proves the optimum is 74.14 k$. The chart rings such points in
  red for exactly this reason.
- **The pruning is lossless.** `test_pruning_is_lossless` checks that `prune=False` reaches the
  same optimal cost as `prune=True` across 8 instances. The bounds cut only branches that provably
  cannot improve the incumbent.

Greedy is strictly costlier than the optimum on 8 of 10 seeds, by up to 62%.

**[`docs/np-hardness.md`](docs/np-hardness.md)** works through why: the reduction from SET-COVER,
why the hardness survives being restricted to disc coverage, Chvatal's H(d) bound with proof
sketch, and the Feige / Dinur-Steurer result that puts greedy near the theoretical ceiling for
any polynomial-time algorithm.

## Tests

```
pytest -q                      # everything (46 tests, ~4 s)
pytest tests/test_planner.py    # algorithms only, under 1 s
```

"It runs" is not evidence for hand-written algorithms, so each one is checked against an
independent oracle:

| Algorithm | Oracle |
|---|---|
| `components`, `largest_component` | `nx.connected_components` (the partition, not just the count) |
| `dijkstra` | `nx.dijkstra_path_length`, plus the path is re-walked and its lengths re-summed |
| `backbone` | `nx.minimum_spanning_tree` on the same metric closure, built from our own distances |
| `edmonds_karp` | `nx.maximum_flow_value` over 300 random networks, plus CLRS fig. 26.1, plus max-flow = min-cut |
| `knapsack` | exhaustive subset search |
| `exact_cover` | exhaustive subset search, and the unpruned search |
| `greedy_cover` | its own (ln n + 1) guarantee, and a hand-built instance that separates cost-weighted from unweighted selection |

`tests/test_app.py` runs the dashboard headlessly through Streamlit's `AppTest`, so a crash in the
sidebar, metric row, Folium map or min-cut table fails a test instead of surfacing as a red box
during the demo.

The suite was validated by mutation testing — ten deliberate bugs were injected into `planner.py`
(reversed Prim comparison, dropped residual reverse edge, off-by-one in the knapsack budget, a
broadcast stage-2 capacity, and so on) and **all ten were caught**. Three of them survived the
first draft of the suite; the tests that now catch them were added in response, including a
randomly-discovered 6-node network where omitting the residual reverse edge silently returns
flow 7 instead of 9.

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

## OpenStreetMap: read this before demoing it

`osm_graph` tries two strategies. `graph_from_place` works when the name geocodes to a real
boundary **polygon** (a city or district). Most *neighbourhood* names come back from Nominatim as
a bare **point**, which `graph_from_place` rejects outright - all four Pune neighbourhoods tested
(Karve Nagar, Shivajinagar, Deccan Gymkhana, Aundh) failed exactly this way. So it falls back to
`graph_from_address` with the sidebar radius, which geocodes to a point and takes a distance. That
radius also bounds the download: an unbounded `"Pune, Maharashtra, India"` ran past two minutes
and would hang a live demo.

**The fetch itself is unverified.** OSM has two services behind it - Nominatim for geocoding,
Overpass for road data - and from the machine this was developed on, Overpass took 15 s just to
answer a status ping and never returned a graph. Everything *around* the fetch is tested offline
(`test_flatten_projected_*`, `test_osm_graph_*`): the lat/lon swap, the parallel-edge collapse,
the self-loop drop, the fallback order, and the error message. **Run it once on your own
connection before relying on it in the demo.** If it fails, the dashboard now shows what each
strategy said instead of a traceback, and the synthetic grid always works offline.

## Still to do

- **A\* for stage 5**, to give the plan's "Dijkstra / A\*" comparison a second data point.
- **Verify the OSM fetch end to end** on a connection with working Overpass access (above).
- **The final report itself.** `docs/np-hardness.md` covers milestone 7's theory half and
  `bench/` holds the runtime curves, but the report document has not been written.
