// Ising kernel: model storage, energy evaluation, simulated annealing.
//
// H(s) = sum_i h_i s_i + sum_{(i,j) in edges} J_ij s_i s_j,   s_i in {-1, +1}

// The beta-validation checks use negated comparisons like `!(beta > 0.0)`
// deliberately: unlike `beta <= 0.0`, the negated form also rejects NaN
// (NaN > 0.0 is false), which is exactly what we want for input validation.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
// The pyo3 0.22 `#[pyfunction]`/`#[pymethods]` macros expand to an error
// conversion (`.into()` into `PyErr`) on every function that returns
// `PyResult`. Since our error type is already `PyErr`, clippy flags it as a
// useless conversion against our `-> PyResult<...>` spans. The `.into()` lives
// in macro-generated code, not our source, so this is a false positive.
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
struct IsingModel {
    #[pyo3(get)]
    num_spins: usize,
    h: Vec<f64>,
    // For each spin i, the edges (j, J_ij) with i < j. Stored once per edge.
    edges_upper: Vec<Vec<(usize, f64)>>,
    // Full neighbor lists (both directions) for fast local-field lookup.
    neighbors: Vec<Vec<(usize, f64)>>,
}

#[pymethods]
impl IsingModel {
    #[new]
    fn new(num_spins: usize, h: Vec<f64>, couplings: Vec<(usize, usize, f64)>) -> PyResult<Self> {
        if h.len() != num_spins {
            return Err(PyValueError::new_err(format!(
                "h has length {} but num_spins is {}",
                h.len(),
                num_spins
            )));
        }
        let mut edges_upper: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_spins];
        let mut neighbors: Vec<Vec<(usize, f64)>> = vec![Vec::new(); num_spins];
        for (i, j, w) in couplings {
            if i >= num_spins || j >= num_spins {
                return Err(PyValueError::new_err("coupling index out of bounds"));
            }
            if i == j {
                return Err(PyValueError::new_err(
                    "self-coupling (i == j) not allowed; put diagonal terms in h",
                ));
            }
            let (a, b) = if i < j { (i, j) } else { (j, i) };
            edges_upper[a].push((b, w));
            neighbors[a].push((b, w));
            neighbors[b].push((a, w));
        }
        Ok(IsingModel { num_spins, h, edges_upper, neighbors })
    }

    fn energy(&self, state: Vec<i64>) -> PyResult<f64> {
        if state.len() != self.num_spins {
            return Err(PyValueError::new_err("state length mismatch"));
        }
        for &s in &state {
            if s != 1 && s != -1 {
                return Err(PyValueError::new_err("state entries must be +1 or -1"));
            }
        }
        Ok(compute_energy(self, &state))
    }

    fn h(&self) -> Vec<f64> {
        self.h.clone()
    }

    fn couplings(&self) -> Vec<(usize, usize, f64)> {
        let mut out = Vec::new();
        for (i, row) in self.edges_upper.iter().enumerate() {
            for &(j, w) in row {
                out.push((i, j, w));
            }
        }
        out
    }
}

fn compute_energy(model: &IsingModel, state: &[i64]) -> f64 {
    let mut e = 0.0;
    for i in 0..model.num_spins {
        e += model.h[i] * state[i] as f64;
        for &(j, w) in &model.edges_upper[i] {
            e += w * state[i] as f64 * state[j] as f64;
        }
    }
    e
}

fn anneal_one(
    model: &IsingModel,
    num_sweeps: usize,
    beta_start: f64,
    beta_end: f64,
    rng: &mut SmallRng,
) -> (Vec<i64>, f64) {
    let n = model.num_spins;
    let mut state: Vec<i64> = (0..n)
        .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
        .collect();

    // Geometric (linear-in-log) beta schedule.
    let log_b0 = beta_start.ln();
    let log_b1 = beta_end.ln();
    let denom = num_sweeps.max(1) as f64;

    for sweep in 0..num_sweeps {
        let t = sweep as f64 / denom;
        let beta = (log_b0 + (log_b1 - log_b0) * t).exp();
        for i in 0..n {
            let s_i = state[i] as f64;
            let mut field = model.h[i];
            for &(j, w) in &model.neighbors[i] {
                field += w * state[j] as f64;
            }
            // Flipping s_i changes energy by dE = -2 * s_i * field.
            let de = -2.0 * s_i * field;
            if de <= 0.0 || rng.gen::<f64>() < (-beta * de).exp() {
                state[i] = -state[i];
            }
        }
    }

    let energy = compute_energy(model, &state);
    (state, energy)
}

#[pyfunction]
#[pyo3(signature = (model, num_sweeps=1000, num_reads=1, beta_start=0.1, beta_end=10.0, seed=None))]
fn simulated_anneal(
    py: Python<'_>,
    model: &IsingModel,
    num_sweeps: usize,
    num_reads: usize,
    beta_start: f64,
    beta_end: f64,
    seed: Option<u64>,
) -> PyResult<Vec<(Vec<i64>, f64)>> {
    // `!(x > 0.0)` rejects NaN as well as non-positive values; a plain
    // `<= 0.0` check would let NaN through (NaN > 0.0 is false).
    if !(beta_start > 0.0) || !(beta_end > 0.0) {
        return Err(PyValueError::new_err("beta_start and beta_end must be > 0"));
    }
    // Pre-generate per-read seeds deterministically; this fixes results across
    // any thread scheduling and means rng draws don't share state between reads.
    let mut master = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };
    let chain_seeds: Vec<u64> = (0..num_reads).map(|_| master.gen()).collect();
    let model = model.clone();

    let results = py.allow_threads(|| {
        chain_seeds
            .into_par_iter()
            .map(|s| {
                let mut rng = SmallRng::seed_from_u64(s);
                anneal_one(&model, num_sweeps, beta_start, beta_end, &mut rng)
            })
            .collect()
    });
    Ok(results)
}

/// Result of one PT chain.
///   best_state, best_energy,
///   swap_acceptance_per_pair,                (length R-1)
///   final_energy_per_position,               (length R, ordered hot -> cold)
///   round_trips_per_replica_id,              (length R, # of hot<->cold label flips)
///   n_up_per_position,                       (length R, KTHT directional flux)
///   n_down_per_position,                     (length R, KTHT directional flux)
///
/// `round_trips[id]` counts how many times replica `id` (the one that *started*
/// at position id) reached an extreme it had not most recently visited -- i.e.
/// label changes. Two label changes form one full out-and-back round trip,
/// so `round_trips[id] / 2` is the canonical "complete round trip" count.
///
/// `n_up[k]` and `n_down[k]` count swap-step samples at position k that hold an
/// "up" replica (last visited the hot end, heading toward cold) or "down"
/// replica (last visited cold, heading toward hot). Replicas with no extreme
/// visited yet are excluded. These are inputs to the KTHT
/// (Katzgraber-Trebst-Huse-Troyer) feedback-optimized beta-ladder tuner.
type PtResult = (Vec<i64>, f64, Vec<f64>, Vec<f64>, Vec<u64>, Vec<u64>, Vec<u64>);

fn pt_one(
    model: &IsingModel,
    betas: &[f64],
    num_sweeps: usize,
    swap_every: usize,
    rng: &mut SmallRng,
) -> PtResult {
    let n = model.num_spins;
    let r = betas.len();

    let mut states: Vec<Vec<i64>> = (0..r)
        .map(|_| {
            (0..n)
                .map(|_| if rng.gen::<bool>() { 1 } else { -1 })
                .collect()
        })
        .collect();
    let mut energies: Vec<f64> = states.iter().map(|s| compute_energy(model, s)).collect();

    let mut best_idx = 0;
    for k in 1..r {
        if energies[k] < energies[best_idx] {
            best_idx = k;
        }
    }
    let mut best_state = states[best_idx].clone();
    let mut best_energy = energies[best_idx];

    let pair_count = r.saturating_sub(1);
    let mut swap_attempts: Vec<u64> = vec![0; pair_count];
    let mut swap_accepts: Vec<u64> = vec![0; pair_count];

    // Round-trip tracking: each "replica" is identified by its starting ladder
    // position; replica_id_at_pos[k] gives the id currently at position k.
    let mut replica_id_at_pos: Vec<usize> = (0..r).collect();
    // last_extreme[id]: 0 = none yet, 1 = last hit hot end (pos 0), 2 = last hit cold end (pos r-1).
    let mut last_extreme: Vec<u8> = vec![0; r];
    let mut round_trips: Vec<u64> = vec![0; r];
    // KTHT directional flux: n_up[k] = swap-step samples where position k holds a
    // replica with last_extreme == 1 (came from hot, heading up to cold).
    // n_down[k] = same with last_extreme == 2 (came from cold, heading down to hot).
    let mut n_up: Vec<u64> = vec![0; r];
    let mut n_down: Vec<u64> = vec![0; r];

    for sweep in 0..num_sweeps {
        for k in 0..r {
            let beta = betas[k];
            for i in 0..n {
                let s_i = states[k][i] as f64;
                let mut field = model.h[i];
                for &(j, w) in &model.neighbors[i] {
                    field += w * states[k][j] as f64;
                }
                let de = -2.0 * s_i * field;
                if de <= 0.0 || rng.gen::<f64>() < (-beta * de).exp() {
                    states[k][i] = -states[k][i];
                    energies[k] += de;
                }
            }
            if energies[k] < best_energy {
                best_energy = energies[k];
                best_state.copy_from_slice(&states[k]);
            }
        }

        if (sweep + 1) % swap_every == 0 {
            // Alternate even/odd pair parity so every pair gets exchange chances.
            let start = sweep % 2;
            let mut k = start;
            while k + 1 < r {
                swap_attempts[k] += 1;
                // Swap accepted with min(1, exp((beta_{k+1} - beta_k)(E_k - E_{k+1}))).
                let delta = (betas[k + 1] - betas[k]) * (energies[k] - energies[k + 1]);
                if delta >= 0.0 || rng.gen::<f64>() < delta.exp() {
                    swap_accepts[k] += 1;
                    states.swap(k, k + 1);
                    energies.swap(k, k + 1);
                    replica_id_at_pos.swap(k, k + 1);
                }
                k += 2;
            }

            // Round-trip bookkeeping: check whether any replica just landed
            // at an extreme it hadn't most recently visited.
            if r >= 2 {
                let hot_id = replica_id_at_pos[0];
                if last_extreme[hot_id] != 1 {
                    if last_extreme[hot_id] == 2 {
                        round_trips[hot_id] += 1;
                    }
                    last_extreme[hot_id] = 1;
                }
                let cold_id = replica_id_at_pos[r - 1];
                if last_extreme[cold_id] != 2 {
                    if last_extreme[cold_id] == 1 {
                        round_trips[cold_id] += 1;
                    }
                    last_extreme[cold_id] = 2;
                }

                // KTHT flux: snapshot direction labels at every position.
                for k in 0..r {
                    match last_extreme[replica_id_at_pos[k]] {
                        1 => n_up[k] += 1,
                        2 => n_down[k] += 1,
                        _ => {}
                    }
                }
            }
        }
    }

    let swap_rates: Vec<f64> = (0..pair_count)
        .map(|k| {
            if swap_attempts[k] > 0 {
                swap_accepts[k] as f64 / swap_attempts[k] as f64
            } else {
                0.0
            }
        })
        .collect();

    // `best_energy` was tracked incrementally via `energies[k] += de`, which
    // accumulates floating-point drift over many sweeps. Recompute it exactly
    // from `best_state` so the returned energy equals `model.energy(best_state)`.
    let best_energy = compute_energy(model, &best_state);

    (best_state, best_energy, swap_rates, energies, round_trips, n_up, n_down)
}

#[pyfunction]
#[pyo3(signature = (
    model,
    num_sweeps=1000,
    num_replicas=8,
    beta_min=0.1,
    beta_max=10.0,
    swap_every=1,
    num_reads=1,
    seed=None,
))]
#[allow(clippy::too_many_arguments)]
fn parallel_tempering(
    py: Python<'_>,
    model: &IsingModel,
    num_sweeps: usize,
    num_replicas: usize,
    beta_min: f64,
    beta_max: f64,
    swap_every: usize,
    num_reads: usize,
    seed: Option<u64>,
) -> PyResult<Vec<(Vec<i64>, f64)>> {
    if num_replicas < 2 {
        return Err(PyValueError::new_err("num_replicas must be >= 2"));
    }
    // `!(x > y)` rejects NaN as well; a plain `<=` check would let NaN through.
    if !(beta_min > 0.0) || !(beta_max > beta_min) {
        return Err(PyValueError::new_err("require 0 < beta_min < beta_max"));
    }
    if swap_every == 0 {
        return Err(PyValueError::new_err("swap_every must be >= 1"));
    }

    let log_lo = beta_min.ln();
    let log_hi = beta_max.ln();
    let denom = (num_replicas - 1) as f64;
    let betas: Vec<f64> = (0..num_replicas)
        .map(|k| (log_lo + (log_hi - log_lo) * (k as f64 / denom)).exp())
        .collect();

    let mut master = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };
    let chain_seeds: Vec<u64> = (0..num_reads).map(|_| master.gen()).collect();
    let model = model.clone();

    let results: Vec<PtResult> = py.allow_threads(|| {
        chain_seeds
            .into_par_iter()
            .map(|s| {
                let mut rng = SmallRng::seed_from_u64(s);
                pt_one(&model, &betas, num_sweeps, swap_every, &mut rng)
            })
            .collect()
    });
    // Drop diagnostics for the lean API.
    Ok(results.into_iter().map(|(s, e, _, _, _, _, _)| (s, e)).collect())
}

#[pyfunction]
#[pyo3(signature = (
    model,
    num_sweeps=1000,
    num_replicas=8,
    beta_min=0.1,
    beta_max=10.0,
    swap_every=1,
    num_reads=1,
    seed=None,
))]
#[allow(clippy::too_many_arguments)]
fn parallel_tempering_diagnostic(
    py: Python<'_>,
    model: &IsingModel,
    num_sweeps: usize,
    num_replicas: usize,
    beta_min: f64,
    beta_max: f64,
    swap_every: usize,
    num_reads: usize,
    seed: Option<u64>,
) -> PyResult<Vec<PtResult>> {
    if num_replicas < 2 {
        return Err(PyValueError::new_err("num_replicas must be >= 2"));
    }
    // `!(x > y)` rejects NaN as well; a plain `<=` check would let NaN through.
    if !(beta_min > 0.0) || !(beta_max > beta_min) {
        return Err(PyValueError::new_err("require 0 < beta_min < beta_max"));
    }
    if swap_every == 0 {
        return Err(PyValueError::new_err("swap_every must be >= 1"));
    }

    let log_lo = beta_min.ln();
    let log_hi = beta_max.ln();
    let denom = (num_replicas - 1) as f64;
    let betas: Vec<f64> = (0..num_replicas)
        .map(|k| (log_lo + (log_hi - log_lo) * (k as f64 / denom)).exp())
        .collect();

    let mut master = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };
    let chain_seeds: Vec<u64> = (0..num_reads).map(|_| master.gen()).collect();
    let model = model.clone();

    let results: Vec<PtResult> = py.allow_threads(|| {
        chain_seeds
            .into_par_iter()
            .map(|s| {
                let mut rng = SmallRng::seed_from_u64(s);
                pt_one(&model, &betas, num_sweeps, swap_every, &mut rng)
            })
            .collect()
    });
    Ok(results)
}

#[pyfunction]
#[pyo3(signature = (model, betas, num_sweeps=1000, swap_every=1, num_reads=1, seed=None))]
fn parallel_tempering_with_betas(
    py: Python<'_>,
    model: &IsingModel,
    betas: Vec<f64>,
    num_sweeps: usize,
    swap_every: usize,
    num_reads: usize,
    seed: Option<u64>,
) -> PyResult<Vec<PtResult>> {
    if betas.len() < 2 {
        return Err(PyValueError::new_err("betas must have at least 2 entries"));
    }
    for &b in &betas {
        if !(b > 0.0) {
            return Err(PyValueError::new_err("all betas must be > 0"));
        }
    }
    for k in 0..betas.len() - 1 {
        if betas[k] >= betas[k + 1] {
            return Err(PyValueError::new_err("betas must be strictly increasing"));
        }
    }
    if swap_every == 0 {
        return Err(PyValueError::new_err("swap_every must be >= 1"));
    }

    let mut master = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };
    let chain_seeds: Vec<u64> = (0..num_reads).map(|_| master.gen()).collect();
    let model = model.clone();

    let results: Vec<PtResult> = py.allow_threads(|| {
        chain_seeds
            .into_par_iter()
            .map(|s| {
                let mut rng = SmallRng::seed_from_u64(s);
                pt_one(&model, &betas, num_sweeps, swap_every, &mut rng)
            })
            .collect()
    });
    Ok(results)
}

// --- Houdayer isoenergetic cluster moves (ICM) on top of parallel tempering ---
//
// The Houdayer move acts on TWO replicas held at the SAME temperature. It builds
// the subgraph induced on the sites where the two replicas disagree, picks one
// connected component (a "cluster"), and flips that cluster in BOTH replicas.
// Because every site on the cluster boundary *agrees* between the two replicas,
// the energy gained by one replica is exactly lost by the other: with no field
// (h == 0) the joint energy E_a + E_b is conserved, so the move is rejection-
// free. It tunnels through barriers that single-spin flips cannot cross.
//
// Effective on finite-dimensional / sparse graphs (e.g. the 3D Edwards-Anderson
// lattice, which is also the regime of hardware spin-glass annealers). On a
// fully connected graph (SK) the disagreeing sites percolate into one component,
// so the move degenerates into a trivial global swap -- it does nothing useful.

fn metropolis_sweep_lane(
    model: &IsingModel,
    betas: &[f64],
    states: &mut [Vec<i64>],
    energies: &mut [f64],
    rng: &mut SmallRng,
) {
    let n = model.num_spins;
    for k in 0..betas.len() {
        let beta = betas[k];
        for i in 0..n {
            let s_i = states[k][i] as f64;
            let mut field = model.h[i];
            for &(j, w) in &model.neighbors[i] {
                field += w * states[k][j] as f64;
            }
            let de = -2.0 * s_i * field;
            if de <= 0.0 || rng.gen::<f64>() < (-beta * de).exp() {
                states[k][i] = -states[k][i];
                energies[k] += de;
            }
        }
    }
}

fn pt_swap_lane(
    betas: &[f64],
    states: &mut [Vec<i64>],
    energies: &mut [f64],
    parity: usize,
    rng: &mut SmallRng,
) {
    let r = betas.len();
    let mut k = parity;
    while k + 1 < r {
        let delta = (betas[k + 1] - betas[k]) * (energies[k] - energies[k + 1]);
        if delta >= 0.0 || rng.gen::<f64>() < delta.exp() {
            states.swap(k, k + 1);
            energies.swap(k, k + 1);
        }
        k += 2;
    }
}

/// One Houdayer cluster move between two same-temperature replicas. Returns the
/// number of spins flipped (0 if rejected or no disagreement). Updates the spin
/// states and their energies in place.
fn houdayer_cluster_move(
    model: &IsingModel,
    sa: &mut [i64],
    sb: &mut [i64],
    ea: &mut f64,
    eb: &mut f64,
    beta: f64,
    rng: &mut SmallRng,
) -> usize {
    let n = model.num_spins;
    let mut diff_sites: Vec<usize> = Vec::new();
    for i in 0..n {
        if sa[i] != sb[i] {
            diff_sites.push(i);
        }
    }
    if diff_sites.is_empty() {
        return 0;
    }
    // Grow a connected cluster of disagreeing sites from a random seed.
    let seed = diff_sites[rng.gen_range(0..diff_sites.len())];
    let mut in_cluster = vec![false; n];
    in_cluster[seed] = true;
    let mut stack = vec![seed];
    let mut cluster: Vec<usize> = Vec::new();
    while let Some(u) = stack.pop() {
        cluster.push(u);
        for &(v, _w) in &model.neighbors[u] {
            if !in_cluster[v] && sa[v] != sb[v] {
                in_cluster[v] = true;
                stack.push(v);
            }
        }
    }
    // Energy change from flipping the cluster. Only boundary bonds (one endpoint
    // in the cluster, one outside) contribute; interior bonds are unchanged.
    let mut d_a = 0.0;
    let mut d_b = 0.0;
    for &i in &cluster {
        d_a += -2.0 * model.h[i] * sa[i] as f64;
        d_b += -2.0 * model.h[i] * sb[i] as f64;
        for &(j, w) in &model.neighbors[i] {
            if !in_cluster[j] {
                d_a += -2.0 * w * sa[i] as f64 * sa[j] as f64;
                d_b += -2.0 * w * sb[i] as f64 * sb[j] as f64;
            }
        }
    }
    // Both replicas share temperature beta; accept on the joint energy change.
    // With h == 0 the boundary terms cancel (d_a + d_b == 0) and this is always
    // accepted. The Metropolis test keeps the move correct even if h != 0.
    let d_total = d_a + d_b;
    if d_total <= 0.0 || rng.gen::<f64>() < (-beta * d_total).exp() {
        for &i in &cluster {
            sa[i] = -sa[i];
            sb[i] = -sb[i];
        }
        *ea += d_a;
        *eb += d_b;
        cluster.len()
    } else {
        0
    }
}

fn pt_houdayer_one(
    model: &IsingModel,
    betas: &[f64],
    num_sweeps: usize,
    swap_every: usize,
    icm_every: usize,
    rng: &mut SmallRng,
) -> (Vec<i64>, f64) {
    let n = model.num_spins;
    let r = betas.len();

    // Two independent lanes of R replicas. The Houdayer move couples the two
    // lanes at matching temperatures; PT swaps act within each lane.
    let rand_states = |rng: &mut SmallRng| -> Vec<Vec<i64>> {
        (0..r)
            .map(|_| (0..n).map(|_| if rng.gen::<bool>() { 1 } else { -1 }).collect())
            .collect()
    };
    let mut sa = rand_states(rng);
    let mut sb = rand_states(rng);
    let mut ea: Vec<f64> = sa.iter().map(|s| compute_energy(model, s)).collect();
    let mut eb: Vec<f64> = sb.iter().map(|s| compute_energy(model, s)).collect();

    let mut best_state = sa[0].clone();
    let mut best_energy = f64::INFINITY;
    let update_best = |states: &[Vec<i64>], energies: &[f64],
                       best_state: &mut Vec<i64>, best_energy: &mut f64| {
        for k in 0..r {
            if energies[k] < *best_energy {
                *best_energy = energies[k];
                best_state.copy_from_slice(&states[k]);
            }
        }
    };
    update_best(&sa, &ea, &mut best_state, &mut best_energy);
    update_best(&sb, &eb, &mut best_state, &mut best_energy);

    for sweep in 0..num_sweeps {
        metropolis_sweep_lane(model, betas, &mut sa, &mut ea, rng);
        metropolis_sweep_lane(model, betas, &mut sb, &mut eb, rng);
        update_best(&sa, &ea, &mut best_state, &mut best_energy);
        update_best(&sb, &eb, &mut best_state, &mut best_energy);

        if (sweep + 1) % swap_every == 0 {
            let parity = sweep % 2;
            pt_swap_lane(betas, &mut sa, &mut ea, parity, rng);
            pt_swap_lane(betas, &mut sb, &mut eb, parity, rng);
        }

        if (sweep + 1) % icm_every == 0 {
            for k in 0..r {
                houdayer_cluster_move(
                    model, &mut sa[k], &mut sb[k], &mut ea[k], &mut eb[k], betas[k], rng,
                );
            }
            update_best(&sa, &ea, &mut best_state, &mut best_energy);
            update_best(&sb, &eb, &mut best_state, &mut best_energy);
        }
    }

    // Recompute exactly to shed any incremental floating-point drift.
    let best_energy = compute_energy(model, &best_state);
    (best_state, best_energy)
}

#[pyfunction]
#[pyo3(signature = (
    model,
    num_sweeps=1000,
    num_replicas=8,
    beta_min=0.1,
    beta_max=10.0,
    swap_every=1,
    icm_every=10,
    num_reads=1,
    seed=None,
))]
#[allow(clippy::too_many_arguments)]
fn parallel_tempering_houdayer(
    py: Python<'_>,
    model: &IsingModel,
    num_sweeps: usize,
    num_replicas: usize,
    beta_min: f64,
    beta_max: f64,
    swap_every: usize,
    icm_every: usize,
    num_reads: usize,
    seed: Option<u64>,
) -> PyResult<Vec<(Vec<i64>, f64)>> {
    if num_replicas < 2 {
        return Err(PyValueError::new_err("num_replicas must be >= 2"));
    }
    if !(beta_min > 0.0) || !(beta_max > beta_min) {
        return Err(PyValueError::new_err("require 0 < beta_min < beta_max"));
    }
    if swap_every == 0 {
        return Err(PyValueError::new_err("swap_every must be >= 1"));
    }
    if icm_every == 0 {
        return Err(PyValueError::new_err("icm_every must be >= 1"));
    }

    let log_lo = beta_min.ln();
    let log_hi = beta_max.ln();
    let denom = (num_replicas - 1) as f64;
    let betas: Vec<f64> = (0..num_replicas)
        .map(|k| (log_lo + (log_hi - log_lo) * (k as f64 / denom)).exp())
        .collect();

    let mut master = match seed {
        Some(s) => SmallRng::seed_from_u64(s),
        None => SmallRng::from_entropy(),
    };
    let chain_seeds: Vec<u64> = (0..num_reads).map(|_| master.gen()).collect();
    let model = model.clone();

    let results = py.allow_threads(|| {
        chain_seeds
            .into_par_iter()
            .map(|s| {
                let mut rng = SmallRng::seed_from_u64(s);
                pt_houdayer_one(&model, &betas, num_sweeps, swap_every, icm_every, &mut rng)
            })
            .collect()
    });
    Ok(results)
}

fn bits_to_state(bits: u64, n: usize) -> Vec<i64> {
    (0..n).map(|i| if (bits >> i) & 1 == 1 { 1 } else { -1 }).collect()
}

#[pyfunction]
fn brute_force_ground_state(py: Python<'_>, model: &IsingModel) -> PyResult<(Vec<i64>, f64)> {
    let n = model.num_spins;
    if n > 30 {
        return Err(PyValueError::new_err(
            "brute force is impractical for N > 30 (2^N evaluations)",
        ));
    }
    let model = model.clone();
    let (best_bits, best_e) = py.allow_threads(|| {
        let total: u64 = 1u64 << n;
        (0..total)
            .into_par_iter()
            .map(|bits| {
                let state = bits_to_state(bits, n);
                (bits, compute_energy(&model, &state))
            })
            .reduce(
                || (0u64, f64::INFINITY),
                |a, b| if b.1 < a.1 { b } else { a },
            )
    });
    Ok((bits_to_state(best_bits, n), best_e))
}

#[pyfunction]
fn brute_force_min_energy(py: Python<'_>, model: &IsingModel) -> PyResult<f64> {
    Ok(brute_force_ground_state(py, model)?.1)
}

#[pymodule]
fn _kernel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<IsingModel>()?;
    m.add_function(wrap_pyfunction!(simulated_anneal, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_tempering, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_tempering_diagnostic, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_tempering_with_betas, m)?)?;
    m.add_function(wrap_pyfunction!(parallel_tempering_houdayer, m)?)?;
    m.add_function(wrap_pyfunction!(brute_force_min_energy, m)?)?;
    m.add_function(wrap_pyfunction!(brute_force_ground_state, m)?)?;
    Ok(())
}
