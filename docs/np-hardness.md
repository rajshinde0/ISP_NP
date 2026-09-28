# Why Stage 1 is Hard: NP-Hardness, Approximation, and What We Measured

*Milestone 7 writeup. Companion to `bench/stage1_scaling.csv`, which holds the measurements
quoted throughout.*

Stage 1 — choosing the cheapest set of towers that still reaches every customer — is the only
stage of this project that is not solvable in polynomial time. This document says precisely
which problem it is, proves it NP-hard, explains why we nonetheless solve it exactly at the
sizes the demo uses, and states what greedy buys and what it costs.

---

## 1. The problem Stage 1 actually solves

**MIN-COST-COVER.**

- **Instance.** A finite universe $D$ of demand points; a family $S_1,\dots,S_m \subseteq D$ of
  coverage sets, one per candidate tower site; a cost $c_i \in \mathbb{Q}_{\ge 0}$ per site.
- **Feasible solution.** A subset $X \subseteq \{1,\dots,m\}$ with
  $\bigcup_{i \in X} S_i = D$.
- **Objective.** Minimise $\sum_{i \in X} c_i$.

In the code (`planner.py`): $D$ is the demand list, $S_i$ is `C[i]["mask"]` (a bitmask over $D$),
and $c_i$ is `C[i]["cost"]`. `exact_cover` returns an optimal $X$; `greedy_cover` returns an
approximate one.

### A precision worth stating explicitly

Each demand point also carries an integer weight $w_d$ (how many customers sit there). **The
weights do not affect what counts as optimal.** `exact_cover` minimises cost subject to *full*
coverage; the weights appear only in three heuristic places:

| Where | Role |
|---|---|
| `greedy_cover`'s ratio | ranks candidates by weight-covered-per-cost |
| `exact_cover`'s `order` | sorts candidates to find good incumbents early |
| `exact_cover`'s fractional bound | converts remaining weight into a cost lower bound |

So this is **min-cost set cover**, not *maximum coverage under a budget* — a different, also
NP-hard, problem that a reader skimming the code could easily mistake it for. Getting this
backwards in the report would be a real error: maximum coverage has a $1 - 1/e$ greedy
guarantee, which is not the bound that applies here.

---

## 2. The decision version is in NP

**DECIDE$(D, S, c, k)$:** does a feasible $X$ with $\sum_{i \in X} c_i \le k$ exist?

A candidate $X$ is a certificate of size at most $m$. Verifying it takes $O(m \cdot |D|)$: OR the
masks together, compare against the full mask, sum the costs. Polynomial, so
DECIDE $\in$ **NP**.

---

## 3. NP-hardness, by reduction from SET-COVER

**SET-COVER** (Karp, 1972 — one of the original 21 NP-complete problems): given a universe $U$,
a family $\mathcal{F}$ of subsets of $U$, and an integer $k$, is there a subfamily of at most $k$
sets whose union is $U$?

**Reduction.** Given a SET-COVER instance $(U, \mathcal{F}, k)$, build a DECIDE instance:

$$D := U, \qquad S_i := F_i \ \text{ for each } F_i \in \mathcal{F}, \qquad c_i := 1, \qquad
\text{budget} := k.$$

This is an identity map plus assigning unit costs — clearly polynomial time.

**Correctness.** With every $c_i = 1$, the cost of $X$ is exactly $|X|$. So a feasible $X$ with
$\sum_{i\in X} c_i \le k$ exists **iff** a subfamily of at most $k$ sets covers $U$. The two
instances are yes-instances together.

Therefore SET-COVER $\le_p$ DECIDE, so DECIDE is NP-hard; combined with §2 it is **NP-complete**,
and the optimisation problem MIN-COST-COVER is **NP-hard**. $\blacksquare$

Note the direction: we reduce the *known-hard* problem **to** ours. Ours is at least as hard.
Doing it the other way round would prove nothing about our problem's difficulty — a common slip
worth being ready for in the viva.

---

## 4. But our instances are geometric — does the hardness survive?

This is the objection that actually matters, and it is easy to miss.

Our generator never produces an arbitrary set system. `make_instance` builds each $S_i$ as

$$S_i = \{\, d \in D : \operatorname{dist}(\text{site}_i, d) \le r_{\text{tier}(i)} \,\}$$

— a **disc**. Restricting the instance family can make a hard problem easy: set cover is
NP-hard in general but solvable in polynomial time when the sets are intervals on a line. So
§3 does not by itself establish that *our* instances are hard.

They are. Covering a finite set of points in the plane by discs is itself NP-hard — the
classical result is Fowler, Paterson and Tanimoto (1981), *"Optimal packing and covering in the
plane are NP-complete"*, which shows covering points by unit squares (and by unit discs) is
NP-complete. Our version is at least as general: two radii rather than one, and non-uniform
costs. So the hardness is a property of the real problem, not an artifact of letting the solver
accept arbitrary bitmasks.

**Honest caveat, and good viva material.** Geometric covering problems are *easier to
approximate* than general set cover, even though both are NP-hard. Unit-disc cover admits a
PTAS (via shifted-grid or local-search techniques), whereas general set cover does not (§7). So
"why not use a PTAS?" is a fair question. Our answers:

1. Our discs are **not unit** — two tiers with a 2.25× radius ratio — and costs are non-uniform,
   so the standard unit-disc PTAS does not apply off the shelf.
2. At the sizes the demo runs, exact branch and bound finishes in **milliseconds** (§6). A PTAS
   would be slower *and* approximate.

---

## 5. Every other stage is polynomial

Stage 1 is the outlier, which is exactly why the plan of action made it the headline.

| Stage | Problem | Algorithm | Complexity | Class |
|---|---|---|---|---|
| 0 | Connectivity | BFS | $O(V+E)$ | P |
| **1** | **Min-cost cover** | **Backtracking + B&B** | $O(2^{m})$ worst case | **NP-hard** |
| 1 | (approximation) | Cost-weighted greedy | $O(m^2 |D| / 64)$ | P |
| 2 | 0/1 knapsack | Dynamic programming | $O(nB)$ *pseudo-polynomial* | NP-hard in general |
| 3 | Minimum spanning tree | Prim | $O(k^2)$ + $k$ Dijkstras | P |
| 4 | Max-flow / min-cut | Edmonds–Karp | $O(VE^2)$ | P |
| 5 | Shortest path | Dijkstra (binary heap) | $O((V+E)\log V)$ | P |

**Stage 2 deserves a footnote.** 0/1 knapsack is NP-hard too, but our DP runs in $O(nB)$ where
$B$ is the budget — *pseudo*-polynomial, because $B$ is a numeric value whose encoding takes
$\log B$ bits, so $O(nB)$ is exponential in the input *size*. It is fast here only because $B$ is
a small integer (≤ 29 k$). Claiming "we solve an NP-hard problem in polynomial time" in the
report would be wrong; the correct statement is that knapsack is only *weakly* NP-hard and admits
a pseudo-polynomial DP. Set cover, by contrast, is **strongly** NP-hard — no such DP exists for
it unless P = NP.

---

## 6. Why we can still solve Stage 1 exactly

Worst case is $O(2^m)$ and we do not escape it. What we do is make the *typical* case tractable
with three pruning rules in `exact_cover`, all of which only ever discard branches that cannot
contain a strictly better solution:

1. **Greedy warm start.** Seed the incumbent with `greedy_cover`'s answer, so the bound is
   already good at the root instead of $\infty$.
2. **Feasibility cuts.** Abandon a branch when its partial cost already meets the incumbent
   (`cost >= best[0]`), or when the union of all *remaining* candidates cannot finish the cover
   (`(cov | suffix[i]) != full`).
3. **Fractional lower bound.** Among candidates still available, take the best
   cost-per-weight ratio $\rho$; covering the remaining weight $W$ cannot possibly cost less
   than $\rho W$. If $\text{cost} + \rho W \ge$ incumbent, prune. This is a relaxation — it lets
   sets be taken fractionally — so it never over-estimates and never prunes a real optimum.

### What that buys, measured

From `bench/stage1_scaling.csv` (60 demand points, seed 0, 20 s limit). "Naive" is the
**identical recursion with all three rules switched off** (`exact_cover(..., prune=False)`), so
node counts compare directly:

| $m = |C|$ | naive nodes | naive time | B&B nodes | B&B time | ratio |
|---|---|---|---|---|---|
| 8 | 511 | 0.0002 s | 22 | 0.0001 s | 23× |
| 12 | 8,191 | 0.004 s | 38 | 0.0002 s | 216× |
| 16 | 127,935 | 0.052 s | 85 | 0.0005 s | 1,505× |
| 20 | 1,946,815 | 0.850 s | 456 | 0.0030 s | 4,269× |
| 24 | 30,799,999 | 13.13 s | 676 | 0.0040 s | 45,562× |
| 28 | 46,284,591 **(timed out)** | >20 s | 572 | 0.0053 s | — |

Three observations for the report:

- **Naive backtracking multiplies its node count by roughly 16 for every 4 extra candidates.**
  That is $2^4$ — the $2^m$ curve, visible directly in the data.
- **Branch and bound does not blow up** at these sizes: under 6 ms through $m = 28$, and 0.12 s
  at $m = 40$. This is *not* a contradiction of NP-hardness — the worst case is untouched, the
  bounds simply happen to be strong on geometric instances where most candidates are either
  clearly worth taking or clearly dominated.
- **A truncated search is not an optimum, and the table proves it.** At $m = 28$ naive's
  incumbent when the clock ran out was **86.32 k\$**, while B&B *proved* the optimum is
  **74.14 k\$**. This is why timed-out points are ringed in red on the chart and never reported
  as optimal.

---

## 7. What greedy gives up

**Guarantee (Chvátal, 1979).** The cost-weighted greedy heuristic returns a cover of cost at
most $H(d) \cdot \text{OPT}$, where $H(n) = \sum_{k=1}^{n} 1/k \le \ln n + 1$ and $d$ is the size
of the largest set.

*Proof sketch (charging).* When greedy selects a set of cost $c$ that newly covers $t$ elements,
charge $c/t$ to each of those $t$ elements. Greedy's total cost is the sum of all charges. Now
fix any set $S^*$ in an optimal cover and list its elements $e_1,\dots,e_p$ in the order greedy
covered them. At the moment $e_j$ was covered, $S^*$ still had at least $p-j+1$ uncovered
elements, so $S^*$ offered a ratio of at most $c(S^*)/(p-j+1)$ — and greedy chose something at
least as good. Hence $\text{charge}(e_j) \le c(S^*)/(p-j+1)$, and

$$\sum_{j=1}^{p} \text{charge}(e_j) \;\le\; c(S^*) \sum_{j=1}^{p} \frac{1}{p-j+1} \;=\; c(S^*)\,H(p).$$

Summing over the sets of the optimal cover accounts for every element at least once, giving
$\text{greedy} \le H(d) \cdot \text{OPT}$. $\blacksquare$

**The weighted form used here.** Our greedy ranks on *weight* covered per cost, not count per
cost. That is the same algorithm run on an instance where a demand point of weight $w$ is $w$
identical copies — so the bound holds with $d$ = the largest total *weight* of any single set.
`test_greedy_is_never_better_than_exact_and_honours_its_bound` checks exactly this form. (An
earlier draft of that test used $H(|D|)$, the unit-weight bound, which is not guaranteed for
weighted instances — it passed on every seed tried but was not sound.)

**The bound is essentially tight**, so this is not pessimism: there are instances forcing greedy
to $\Omega(\log n) \cdot \text{OPT}$.

**And no polynomial algorithm does better.** Feige (1998) showed set cover admits no polynomial
$(1-\varepsilon)\ln n$ approximation unless $\text{NP} \subseteq \text{DTIME}(n^{O(\log\log n)})$;
Dinur and Steurer (2014) strengthened the assumption to simply $\text{P} \ne \text{NP}$. So
greedy's $\ln n + 1$ is, up to lower-order terms, **the best any polynomial-time algorithm can
guarantee**. That is the real argument for shipping greedy alongside the exact solver — not that
it is a convenient hack, but that it is provably near the theoretical ceiling.

### What we actually observed

Across seeds 0–9 at $m = 20$, greedy was **strictly more expensive than the optimum on 8 of 10
instances**, by up to **62%**. Comfortably inside the $H(d)$ guarantee, but far from free — and a
direct demonstration that the NP-hardness has a *price*, which is the point the plan wanted the
demo to land.

---

## 8. Summary for the report

1. Stage 1 is min-cost set cover: **NP-hard** by reduction from SET-COVER, and still NP-hard
   restricted to the disc-coverage instances we actually generate.
2. Its decision version is **NP-complete**.
3. Exact solution is exponential in the worst case; branch and bound with a greedy warm start,
   feasibility cuts and a fractional lower bound makes it practical to $m = 40$ here — cutting
   the search tree by up to **45,562×** against the same recursion unpruned.
4. Greedy guarantees $H(d) \le \ln n + 1$ times optimal, and by Feige / Dinur–Steurer that is
   essentially optimal for any polynomial-time algorithm.
5. Every other stage is genuinely polynomial, except stage 2's knapsack, which is
   *pseudo*-polynomial and fast only because the budget is a small number.

---

## References

Verify these against your course's citation requirements before submission — the arguments above
stand on their own, but page and edition details are not checked here.

- R. M. Karp (1972). *Reducibility Among Combinatorial Problems.* Complexity of Computer
  Computations. — set cover / minimum cover NP-completeness.
- V. Chvátal (1979). *A Greedy Heuristic for the Set-Covering Problem.* Mathematics of
  Operations Research 4(3). — the $H(n)$ bound for weighted set cover.
- R. J. Fowler, M. S. Paterson, S. L. Tanimoto (1981). *Optimal packing and covering in the
  plane are NP-complete.* Information Processing Letters 12(3). — geometric covering hardness.
- U. Feige (1998). *A Threshold of $\ln n$ for Approximating Set Cover.* Journal of the ACM
  45(4).
- I. Dinur, D. Steurer (2014). *Analytical Approach to Parallel Repetition.* STOC. — $\ln n$
  hardness under P ≠ NP.
- T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein. *Introduction to Algorithms* — greedy
  set cover, Edmonds–Karp, Prim, Dijkstra, knapsack DP.
