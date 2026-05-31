# Positioning vs D-Wave: what the claim is, and where this lab fits

This document is the honest, cited answer to "are we competing with D-Wave?"
It is **not** a head-to-head benchmark against a D-Wave QPU — we have not run one,
and the absolute comparison is blocked on data we do not have (see *What we have
not done*). It is a precise positioning plus a falsifiable hypothesis and a
roadmap to test it.

## 1. D-Wave's strongest *optimization* advantage claim

The most relevant published result is:

> **H. Munoz Bauza and D. A. Lidar, "Scaling Advantage in Approximate
> Optimization with Quantum Annealing," Phys. Rev. Lett. 134, 160601 (2025)**
> (arXiv:2401.07184).

Verbatim from the abstract: *"we present evidence for a quantum annealing scaling
advantage in approximate optimization. The advantage is relative to the top
classical heuristic algorithm: parallel tempering with isoenergetic cluster moves
(PT-ICM)."* The setting is *"a family of 2D spin-glass problems with high-precision
spin-spin interactions"* on a *degree-5 interaction graph* with *"over 1,300
error-suppressed logical qubits"* via quantum annealing correction (QAC). The
metric is **time-to-epsilon** (time-to-solution generalized to low-energy states),
and the advantage is demonstrated *"at sampling low energy states with an
optimality gap of at least 1.0%."*

Two facts matter for us:

1. The advantage is **approximate optimization** (≥1.0% optimality gap), not exact
   ground states; a computational quantum advantage in *exact* optimization
   *"has so far remained elusive."*
2. The classical baseline is **PT-ICM** — which the authors call the *"best
   currently available general heuristic classical optimization method."* The
   advantage *"is diminished without QAC."*

Earlier scaling-advantage claims were weaker baselines: Albash & Lidar
(*Phys. Rev. X 8, 031016, 2018*; arXiv:1705.07452) showed an advantage over
**simulated annealing**; King et al. (*Nat. Commun. 12, 1113, 2021*) showed an
advantage over **path-integral Monte Carlo** but for quantum *simulation* of
dynamics, not optimization.

## 2. Why this lab is positioned against exactly that baseline

PT-ICM is precisely our `parallel_tempering_houdayer` (parallel tempering +
Houdayer isoenergetic cluster moves). And our central, reproducible finding is
that **population annealing dramatically outperforms PT-ICM** on 3D
Edwards-Anderson glasses, measured as optimal work-to-solution in
hardware-independent Monte Carlo sweep units (`results/pa_vs_pticm_tts_ea3d.json`):

| L | N   | PA W\*  | PT-ICM W\*    | PT-ICM / PA |
|---|-----|--------:|--------------:|------------:|
| 4 | 64  | 5,238   | 33,996        | 6×          |
| 5 | 125 | 10,690  | 19,236,902    | **1,800×**  |
| 6 | 216 | 71,229  | *unreached*   | ∞           |

PT-ICM's work-to-solution diverges relative to PA as size grows; by N=216 PT-ICM
fails to reach the best-known energy within budget while PA solves it. (This is
consistent with the population-annealing literature, e.g. Wang, Machta &
Katzgraber, *Phys. Rev. E 92, 063307, 2015*.)

### 2b. The same result on a 2D degree-5 proxy of the D-Wave instance class

The PRL benchmark is *2D, degree-5, high-precision*, not 3D cubic. To test whether
PA's dominance survives a change to that instance *type*, we built a matched-in-
spirit proxy — a 5-regular 2D toroidal lattice (4 nearest neighbors + 1 diagonal
per site) with Gaussian (high-precision) couplings, `degree5_2d_instance`. This is
**not** D-Wave's exact QAC graph, but it shares the salient features. The same
work-to-solution measurement (`results/pa_vs_pticm_degree5_2d.json`):

| L  | N   | PA W\*   | PT-ICM W\*   | PT-ICM / PA |
|----|-----|---------:|-------------:|------------:|
| 8  | 64  | 5,474    | 201,184      | 37×         |
| 10 | 100 | 21,342   | 5,142,975    | 241×        |
| 12 | 144 | 113,071  | *unreached*  | ∞           |
| 14 | 196 | 540,844  | *unreached*  | ∞           |

The divergence is the same as on 3D cubic EA: PA stays tractable while PT-ICM's
work-to-solution explodes and then fails to reach the floor. Notably, Houdayer
cluster moves do **not** rescue PT-ICM here — on this degree-5, high-precision
graph the disagreement clusters percolate (as on SK) and the isoenergetic moves
add little. So PA's advantage over PT-ICM is **not** specific to 3D cubic EA; it
holds on the 2D degree-5 high-precision instance *type* as well.

## 3. The falsifiable hypothesis

D-Wave's strongest optimization-advantage claim is measured against PT-ICM. We
find that PA is a *much* stronger classical optimizer than PT-ICM — on 3D cubic EA
**and** on a 2D degree-5 high-precision proxy of their own instance class.
Therefore:

> **Hypothesis:** a population-annealing baseline (PA, or PA+ICM) would narrow or
> eliminate the QA-vs-PT-ICM scaling advantage reported in PRL 134, 160601 (2025).

This is the standard mechanism by which quantum-advantage claims are tested and
often overturned — a stronger classical baseline. It is *falsifiable*: it predicts
that re-running the PRL benchmark with PA in place of PT-ICM would shrink the
reported advantage factor.

**Crucial logical caveat.** PA ≫ PT-ICM does **not** by itself imply PA ≫ QA. Both
PA and QA beat PT-ICM; whether PA beats QA depends on comparing the two advantage
factors *against the same PT-ICM baseline, on the same instances, in the same
metric.* That common-baseline comparison is the actual test, and we have not done
it — we lack the QA time-to-epsilon data in a form commensurable with our sweep
units. So this remains a motivated hypothesis, not a result.

## 4. What we have NOT done, and cannot claim

Honesty requires stating the gaps plainly. **We have not refuted, or even
directly tested, the D-Wave claim.** Specifically:

- **Proxy instances, not the real ones.** We now have results on both 3D cubic EA
  *and* a 2D degree-5 high-precision proxy, and PA dominates PT-ICM on both. But
  the proxy is our own 5-regular toroidal construction, **not** D-Wave's exact QAC
  graph; instance-class details (the precise connectivity, coupling precision,
  boundary structure) can change the picture and must be matched.
- **Possibly under-tuned PT-ICM.** Our PT-ICM uses *fixed* hyperparameters
  (16 replicas, β ∈ [0.05, 6.0]) scanned only over sweeps. The PRL tuned PT-ICM
  carefully. A better-tuned PT-ICM would shrink the PA/PT-ICM ratios we report —
  the *direction* (PA wins) is robust across our runs, but the *magnitude* is an
  upper bound on PA's true advantage over a well-tuned PT-ICM.
- **No direct PA-vs-QA comparison.** As noted in §3, beating PT-ICM is not beating
  QA; the common-baseline comparison in matched units has not been done.
- **Different units.** Our metric is Monte Carlo **sweep units** (deliberately
  hardware-independent). The PRL metric is **time-to-epsilon in wall-clock**,
  mixing QPU annealing time against classical CPU time. Bridging these honestly
  requires either a wall-clock accounting on matched hardware or a scaling-exponent
  comparison (which is the defensible route — see below).
- **No matched instances.** We have not obtained or regenerated the PRL instance
  set, nor implemented their exact time-to-epsilon protocol or their PT-ICM tuning.
- **Approximate vs exact.** Their advantage is at a ≥1.0% optimality gap; our
  work-to-solution is to the best-known energy. The target must be matched.

Any statement stronger than the hypothesis in §3 would be the exact
"looks-like-a-win-but-isn't" benchmark this project has avoided throughout.

## 5. Roadmap to an honest head-to-head

1. **Match the instances.** Obtain the PRL instance set (or generate random
   spin glasses on the same degree-5, high-precision graph). Sweep size.
2. **Match the metric.** Implement time-to-epsilon at a fixed optimality gap
   (start at 1.0%), with ground-truth / best-known floors per instance.
3. **Match the baseline.** Reproduce PT-ICM at the tuning used in the PRL, verify
   our `parallel_tempering_houdayer` reproduces its scaling, then run PA / PA+ICM
   under the same budget protocol.
4. **Compare scaling exponents, not absolute wall time.** The honest claim is
   about how time-to-epsilon *scales* with size; absolute QPU-vs-CPU wall time
   depends on hardware clocks and is not the point.
5. **Publish the negative case too.** If PA does *not* erode the advantage on the
   matched instances, that strengthens the D-Wave result and is worth reporting.

Until steps 1–4 are done, the correct statement is: *D-Wave's strongest
optimization advantage is over PT-ICM; this lab has a much stronger classical
optimizer (PA) than PT-ICM on 3D spin glasses; whether that closes the gap on
D-Wave's own instance class is an open, testable question.*

## Sources

- Munoz Bauza & Lidar, *Scaling Advantage in Approximate Optimization with Quantum
  Annealing*, Phys. Rev. Lett. 134, 160601 (2025), arXiv:2401.07184.
- Albash & Lidar, *Demonstration of a Scaling Advantage for a Quantum Annealer over
  Simulated Annealing*, Phys. Rev. X 8, 031016 (2018), arXiv:1705.07452.
- King et al., *Scaling advantage over path-integral Monte Carlo in quantum
  simulation of geometrically frustrated magnets*, Nat. Commun. 12, 1113 (2021).
- Rønnow et al., *Defining and detecting quantum speedup*, Science 345, 420 (2014).
- Wang, Machta & Katzgraber, *Population annealing: Theory and application in spin
  glasses*, Phys. Rev. E 92, 063307 (2015).
