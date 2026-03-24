"""
Experiment: Large-N QFT Rank Scaling with Online RREF
=====================================================

Extends the QFT GF(2) rank analysis to N = 128, 256, (512) using an online
RREF approach that avoids storing the full t×N matrix.

Key design:
  - Online RREF: Maintain an N×N matrix in reduced form. For each new T-gate
    x-bit row, try to reduce against existing pivots. If it doesn't reduce to
    zero, add it as a new pivot. O(N²) memory, O(N) per row insertion.
  - Uses the SAME circuit generation (generate_qft_circuit_large) and Clifford
    walk (compute_twisted_pauli + OFD) as the existing experiment_qft_rank_scaling.jl
    to guarantee identical results at overlapping N values.
  - Verification: Must reproduce existing N=4..64 results EXACTLY before running
    large N.

Step 1: Verify against existing data at N=4,6,8,10,12,16,20,24,32,48,64
Step 2: Run N=128, 256, (512 if feasible)
Step 3: Fit three models (power law, log-corrected linear, log-corrected power)
Step 4: Compute BIC for model selection

Usage:
    julia benchmarks/experiment_qft_large_n.jl [mode]

Modes:
    "verify"  - Only run verification against existing data
    "medium"  - Verify + run N=128, 256
    "full"    - Verify + run N=128, 256, 512
"""

camps_dir = dirname(dirname(@__FILE__))
println("Activating CAMPS.jl environment: $camps_dir")
using Pkg
Pkg.activate(camps_dir)

using CAMPS
using QuantumClifford
using QuantumClifford.ECC
using Random
using Statistics
using Printf
using Dates

"""
    generate_qft_circuit_large(n_qubits; density=:medium, seed=42)

Generate QFT circuit with exact-angle Eq. 3 decomposition at arbitrary qubit count.
Same decomposition logic as QFTFamily but without the n≤8 restriction.

The QFT on n qubits consists of:
- n Hadamard gates
- n(n-1)/2 controlled-R_k gates (k=2..n)
- Each controlled-R_k decomposes via Eq. 3 into 3 non-Clifford Rz gates
  with exact angle θ = 2π/2^k

density controls the maximum k value:
- :low → k_max = n+1 (all rotations, full QFT)
- :medium → k_max = 6
- :high → k_max = 4 (fewest rotations, most approximate)
"""
function generate_qft_circuit_large(n_qubits::Int; density::Symbol=:medium, seed::Int=42)
    density in (:low, :medium, :high) || throw(ArgumentError("density must be :low, :medium, or :high"))

    rng = Random.MersenneTwister(seed)

    k_max = density == :low ? n_qubits + 1 : (density == :medium ? 6 : 4)

    gates = Gate[]
    t_positions = Int[]

    for j in 1:n_qubits
        push!(gates, CliffordGate([(:H, j)], [j]))

        for k in 2:(n_qubits - j + 1)
            k > k_max && break

            control_qubit = j + k - 1
            target_qubit = j

            θ = 2π / 2^k
            push!(gates, RotationGate(target_qubit, :Z, θ/2))
            push!(t_positions, length(gates))
            push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
            push!(gates, RotationGate(target_qubit, :Z, -θ/2))
            push!(t_positions, length(gates))
            push!(gates, CliffordGate([(:CNOT, control_qubit, target_qubit)], [control_qubit, target_qubit]))
            push!(gates, RotationGate(control_qubit, :Z, θ/2))
            push!(t_positions, length(gates))
        end
    end

    n_swaps = div(n_qubits, 2)
    for i in 1:n_swaps
        qubit1 = i
        qubit2 = n_qubits - i + 1
        push!(gates, CliffordGate([(:CNOT, qubit1, qubit2)], [qubit1, qubit2]))
        push!(gates, CliffordGate([(:CNOT, qubit2, qubit1)], [qubit2, qubit1]))
        push!(gates, CliffordGate([(:CNOT, qubit1, qubit2)], [qubit1, qubit2]))
    end

    return (n_qubits=n_qubits, gates=gates, t_positions=t_positions,
            metadata=Dict{String,Any}("family"=>"QFT", "density"=>string(density),
                                       "k_max"=>k_max, "n_t_gates"=>length(t_positions)))
end

"""
    OnlineRREF

Maintains an N×N binary matrix in reduced row echelon form (RREF) for
streaming GF(2) rank computation.

Memory: O(N²) regardless of how many rows are inserted.
Cost per insertion: O(N × current_rank) worst case.
"""
mutable struct OnlineRREF
    n::Int
    matrix::Matrix{Bool}
    pivot_col::Vector{Int}
    rank::Int
end

"""
    OnlineRREF(n::Int) -> OnlineRREF

Create an empty online RREF structure for n-column binary vectors.
"""
function OnlineRREF(n::Int)
    return OnlineRREF(
        n,
        zeros(Bool, n, n),
        zeros(Int, n),
        0
    )
end

"""
    try_insert!(rref::OnlineRREF, row::BitVector) -> Bool

Try to insert a new row into the online RREF structure.

Returns true if the row was linearly independent (rank increased by 1),
false if it was linearly dependent (rank unchanged).

Algorithm:
1. Copy the row and reduce it against existing pivots by XOR
2. If it reduces to zero → linearly dependent → return false
3. If not → find first nonzero bit as new pivot → add to RREF → return true
"""
function try_insert!(rref::OnlineRREF, row::BitVector)::Bool
    @assert length(row) == rref.n "Row length $(length(row)) != n=$(rref.n)"

    r = copy(row)
    for i in 1:rref.rank
        pc = rref.pivot_col[i]
        if r[pc]
            for j in 1:rref.n
                r[j] = r[j] ⊻ rref.matrix[i, j]
            end
        end
    end

    new_pivot = findfirst(r)
    if new_pivot === nothing
        return false
    end

    rref.rank += 1
    for j in 1:rref.n
        rref.matrix[rref.rank, j] = r[j]
    end
    rref.pivot_col[rref.rank] = new_pivot

    return true
end

"""
    qft_gf2_rank_online(n_qubits::Int; seed::Int=42) -> NamedTuple

Compute GF(2) rank of QFT circuit using online RREF.

Uses the exact same circuit generation and Clifford walk as the existing
experiment_qft_rank_scaling.jl, but replaces the full-matrix gf2_rank()
with streaming online RREF.

This is equivalent to:
  1. generate_qft_circuit_large(n; density=:low, seed=seed)
  2. Walk through gates with CAMPSState
  3. At each T-gate: compute twisted Pauli, extract x-bits, try_insert! into RREF
  4. Apply OFD if possible (to match existing behavior)

Returns: (n_qubits, seed, t, rank, nullity, predicted_chi, nullity_ratio)
"""
function qft_gf2_rank_online(n_qubits::Int; seed::Int=42)
    circuit = generate_qft_circuit_large(n_qubits; density=:low, seed=seed)
    gates = circuit.gates

    state = CAMPSState(n_qubits)
    initialize!(state)

    rref = OnlineRREF(n_qubits)
    t_count = 0

    for gate in gates
        if gate isa CliffordGate
            apply_clifford_gate_to_state!(state, gate)

        elseif gate isa RotationGate
            if is_clifford_angle(gate.angle)
                clifford_gate = rotation_to_clifford(gate)
                if clifford_gate !== nothing
                    apply_clifford_gate_to_state!(state, clifford_gate)
                end
            else
                t_count += 1

                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)

                xb = BitVector(xbit(P_twisted))

                try_insert!(rref, xb)

                control = find_disentangling_qubit(P_twisted, state.free_qubits)
                if control !== nothing
                    D_gates = build_disentangler_gates(P_twisted, control)
                    D_flat = flatten_gate_sequence(D_gates)
                    apply_inverse_gates!(state.clifford, D_flat)
                    state.free_qubits[control] = false
                end
            end
        end
    end

    nullity = t_count - rref.rank
    predicted_chi = nullity >= 62 ? (typemax(Int) ÷ 2) : (nullity <= 0 ? 1 : 2^nullity)
    nullity_ratio = t_count > 0 ? nullity / t_count : NaN

    return (
        n_qubits = n_qubits,
        seed = seed,
        t = t_count,
        rank = rref.rank,
        nullity = nullity,
        predicted_chi = predicted_chi,
        nullity_ratio = nullity_ratio
    )
end

const KNOWN_QFT_RESULTS = Dict(
    4  => (t=16,     rank=2),
    6  => (t=80,     rank=4),
    8  => (t=224,    rank=6),
    10 => (t=480,    rank=8),
    12 => (t=880,    rank=10),
    16 => (t=2240,   rank=14),
    20 => (t=4560,   rank=18),
    24 => (t=8096,   rank=22),
    32 => (t=19840,  rank=30),
    48 => (t=69184,  rank=46),
    64 => (t=166656, rank=62),
)

"""
    run_verification() -> Bool

Verify that the online RREF approach reproduces existing results EXACTLY.
Returns true if all verifications pass, false otherwise.
"""
function run_verification()
    println("="^80)
    println("STEP 1: VERIFICATION — Online RREF vs Known Results")
    println("="^80)
    println()

    all_pass = true
    n_range = sort(collect(keys(KNOWN_QFT_RESULTS)))

    @printf("%6s %10s %10s %10s %10s %8s\n",
            "N", "t_expect", "t_got", "rank_exp", "rank_got", "Status")
    println("-"^60)

    for n in n_range
        known = KNOWN_QFT_RESULTS[n]
        seed = Int(hash(("qft_scaling", n, 1)) % UInt32)

        t_start = time()
        result = qft_gf2_rank_online(n; seed=seed)
        elapsed = time() - t_start

        t_match = result.t == known.t
        rank_match = result.rank == known.rank
        pass = t_match && rank_match

        status = pass ? "PASS" : "FAIL"
        if !pass
            all_pass = false
        end

        @printf("%6d %10d %10d %10d %10d %8s  (%.1fs)\n",
                n, known.t, result.t, known.rank, result.rank, status, elapsed)
    end

    println("-"^60)
    if all_pass
        println("ALL VERIFICATIONS PASSED — Online RREF matches existing results exactly.")
    else
        println("VERIFICATION FAILED — Results do not match. DO NOT proceed to large N.")
    end
    println()

    return all_pass
end

"""
    run_large_n(n_range::Vector{Int}) -> Vector{NamedTuple}

Run GF(2) rank analysis for large N values.
"""
function run_large_n(n_range::Vector{Int})
    println("="^80)
    println("STEP 2: LARGE-N QFT RANK SCALING")
    println("="^80)
    println()

    results = NamedTuple[]

    for n in n_range
        seed = Int(hash(("qft_scaling", n, 1)) % UInt32)

        println("Running N=$n ...")
        t_start = time()
        result = qft_gf2_rank_online(n; seed=seed)
        elapsed = time() - t_start

        push!(results, result)

        @printf("  N=%d: t=%d, rank=%d, nullity=%d, ν/t=%.6f  (%.1fs)\n",
                n, result.t, result.rank, result.nullity,
                isnan(result.nullity_ratio) ? 0.0 : result.nullity_ratio, elapsed)
    end

    return results
end

"""
    fit_power_law(ns, ranks) -> (a, b, r_sq, rss)

Fit rank = a · N^b via linear regression on log-log.
Returns (a, b, r_squared, residual_sum_of_squares).
"""
function fit_power_law_extended(ns::Vector{Float64}, ranks::Vector{Float64})
    valid = [(n, r) for (n, r) in zip(ns, ranks) if r > 0 && n > 0]
    isempty(valid) && return (NaN, NaN, NaN, NaN)

    log_ns = [log(v[1]) for v in valid]
    log_ranks = [log(v[2]) for v in valid]

    npts = length(log_ns)
    sum_x = sum(log_ns)
    sum_y = sum(log_ranks)
    sum_xy = sum(log_ns .* log_ranks)
    sum_x2 = sum(log_ns .^ 2)

    denom = npts * sum_x2 - sum_x^2
    abs(denom) < 1e-12 && return (NaN, NaN, NaN, NaN)

    slope = (npts * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / npts

    a = exp(intercept)
    b = slope

    predicted = intercept .+ slope .* log_ns
    residuals = log_ranks .- predicted
    rss = sum(residuals .^ 2)
    mean_y = sum_y / npts
    ss_tot = sum((log_ranks .- mean_y).^2)
    r_sq = ss_tot > 0 ? 1 - rss / ss_tot : NaN

    return (a, b, r_sq, rss)
end

"""
    fit_log_corrected_linear(ns, ranks) -> (c, a_exp, r_sq, rss)

Fit rank = c · N · log^a(N) via nonlinear least squares on log(rank).
In log space: log(rank) = log(c) + log(N) + a·log(log(N))
This is linear in (log(c), a) with known coefficient 1 for log(N).
"""
function fit_log_corrected_linear(ns::Vector{Float64}, ranks::Vector{Float64})
    valid = [(n, r) for (n, r) in zip(ns, ranks) if r > 0 && n > 1]
    isempty(valid) && return (NaN, NaN, NaN, NaN)

    log_ranks = [log(v[2]) for v in valid]
    log_ns = [log(v[1]) for v in valid]
    loglog_ns = [log(log(v[1])) for v in valid]

    npts = length(log_ranks)

    y_adj = log_ranks .- log_ns

    sum_x = sum(loglog_ns)
    sum_y = sum(y_adj)
    sum_xy = sum(loglog_ns .* y_adj)
    sum_x2 = sum(loglog_ns .^ 2)

    denom = npts * sum_x2 - sum_x^2
    abs(denom) < 1e-12 && return (NaN, NaN, NaN, NaN)

    a_exp = (npts * sum_xy - sum_x * sum_y) / denom
    log_c = (sum_y - a_exp * sum_x) / npts
    c = exp(log_c)

    predicted = log_c .+ log_ns .+ a_exp .* loglog_ns
    residuals = log_ranks .- predicted
    rss = sum(residuals .^ 2)
    mean_y = sum(log_ranks) / npts
    ss_tot = sum((log_ranks .- mean_y).^2)
    r_sq = ss_tot > 0 ? 1 - rss / ss_tot : NaN

    return (c, a_exp, r_sq, rss)
end

"""
    fit_log_corrected_power(ns, ranks) -> (c, alpha, a_exp, r_sq, rss)

Fit rank = c · N^α · log^a(N) via grid search + linear regression.
In log space: log(rank) = log(c) + α·log(N) + a·log(log(N))
Linear in (log(c), α, a).
"""
function fit_log_corrected_power(ns::Vector{Float64}, ranks::Vector{Float64})
    valid = [(n, r) for (n, r) in zip(ns, ranks) if r > 0 && n > 1]
    isempty(valid) && return (NaN, NaN, NaN, NaN, NaN)

    log_ranks = [log(v[2]) for v in valid]
    log_ns = [log(v[1]) for v in valid]
    loglog_ns = [log(log(v[1])) for v in valid]

    npts = length(log_ranks)
    npts < 3 && return (NaN, NaN, NaN, NaN, NaN)

    X = hcat(ones(npts), log_ns, loglog_ns)
    Y = log_ranks

    XtX = X' * X
    XtY = X' * Y

    det_XtX = XtX[1,1] * (XtX[2,2]*XtX[3,3] - XtX[2,3]*XtX[3,2]) -
              XtX[1,2] * (XtX[2,1]*XtX[3,3] - XtX[2,3]*XtX[3,1]) +
              XtX[1,3] * (XtX[2,1]*XtX[3,2] - XtX[2,2]*XtX[3,1])

    abs(det_XtX) < 1e-12 && return (NaN, NaN, NaN, NaN, NaN)

    beta = XtX \ XtY

    log_c = beta[1]
    alpha = beta[2]
    a_exp = beta[3]
    c = exp(log_c)

    predicted = X * beta
    residuals = Y .- predicted
    rss = sum(residuals .^ 2)
    mean_y = sum(Y) / npts
    ss_tot = sum((Y .- mean_y).^2)
    r_sq = ss_tot > 0 ? 1 - rss / ss_tot : NaN

    return (c, alpha, a_exp, r_sq, rss)
end

"""
    compute_bic(n_points::Int, n_params::Int, rss::Float64) -> Float64

Compute Bayesian Information Criterion.
BIC = n·ln(RSS/n) + k·ln(n)
"""
function compute_bic(n_points::Int, n_params::Int, rss::Float64)::Float64
    n_points <= 0 && return NaN
    rss <= 0 && return -Inf
    return n_points * log(rss / n_points) + n_params * log(n_points)
end

"""
    run_model_fitting(ns, ranks) -> NamedTuple

Fit three models and compute BIC for model selection.
"""
function run_model_fitting(ns::Vector{Float64}, ranks::Vector{Float64})
    println("="^80)
    println("STEP 3: MODEL FITTING AND SELECTION")
    println("="^80)
    println()

    n_points = length(ns)

    a1, b1, r2_1, rss1 = fit_power_law_extended(ns, ranks)
    bic1 = compute_bic(n_points, 2, rss1)
    @printf("Model 1: rank = %.4f · N^%.4f\n", a1, b1)
    @printf("         R² = %.6f, RSS = %.6e, BIC = %.2f\n\n", r2_1, rss1, bic1)

    c2, a2, r2_2, rss2 = fit_log_corrected_linear(ns, ranks)
    bic2 = compute_bic(n_points, 2, rss2)
    @printf("Model 2: rank = %.4f · N · log(N)^%.4f\n", c2, a2)
    @printf("         R² = %.6f, RSS = %.6e, BIC = %.2f\n\n", r2_2, rss2, bic2)

    c3, a3, e3, r2_3, rss3 = fit_log_corrected_power(ns, ranks)
    bic3 = compute_bic(n_points, 3, rss3)
    @printf("Model 3: rank = %.4f · N^%.4f · log(N)^%.4f\n", c3, a3, e3)
    @printf("         R² = %.6f, RSS = %.6e, BIC = %.2f\n\n", r2_3, rss3, bic3)

    bics = [bic1, bic2, bic3]
    model_names = ["Pure power law", "Log-corrected linear", "Log-corrected power"]
    best_idx = argmin(bics)
    best_bic = bics[best_idx]

    println("-"^60)
    println("MODEL SELECTION (lower BIC is better):")
    println("-"^60)
    for i in 1:3
        marker = i == best_idx ? " ← BEST" : ""
        delta = bics[i] - best_bic
        @printf("  %s: BIC = %.2f (ΔBIC = %.2f)%s\n",
                model_names[i], bics[i], delta, marker)
    end
    println()

    delta_bic_12 = abs(bic1 - bic2)
    delta_bic_best_second = sort(bics)[2] - best_bic
    if delta_bic_best_second > 10
        println("Strong evidence (ΔBIC > 10) for $(model_names[best_idx]).")
    elseif delta_bic_best_second > 6
        println("Moderate evidence (ΔBIC > 6) for $(model_names[best_idx]).")
    elseif delta_bic_best_second > 2
        println("Weak evidence (ΔBIC > 2) for $(model_names[best_idx]).")
    else
        println("Models are not clearly distinguishable (ΔBIC < 2).")
    end
    println()

    return (
        model1 = (a=a1, b=b1, r_sq=r2_1, rss=rss1, bic=bic1),
        model2 = (c=c2, a=a2, r_sq=r2_2, rss=rss2, bic=bic2),
        model3 = (c=c3, alpha=a3, a=e3, r_sq=r2_3, rss=rss3, bic=bic3),
        best_model = best_idx,
        delta_bic = delta_bic_best_second
    )
end

"""
    compute_effective_exponents(ns, ranks) -> Vector{NamedTuple}

Compute local power-law exponents between consecutive data points.
α_eff(i) = log(rank[i+1]/rank[i]) / log(N[i+1]/N[i])

This reveals whether the exponent is converging (α → 1) or stable (α ≈ const).
"""
function compute_effective_exponents(ns::Vector{Float64}, ranks::Vector{Float64})
    exponents = NamedTuple[]
    for i in 1:(length(ns)-1)
        if ranks[i] > 0 && ranks[i+1] > 0 && ns[i] > 0 && ns[i+1] > 0
            alpha_eff = log(ranks[i+1] / ranks[i]) / log(ns[i+1] / ns[i])
            push!(exponents, (
                n_low = ns[i],
                n_high = ns[i+1],
                alpha_eff = alpha_eff
            ))
        end
    end
    return exponents
end

function write_results_csv(results::Vector, output_path::String)
    open(output_path, "w") do io
        println(io, "success,n_qubits,seed,t,rank,nullity,predicted_chi,nullity_ratio")
        for r in results
            chi_str = r.predicted_chi >= typemax(Int) ÷ 2 ? "$(typemax(Int) ÷ 2)" : "$(r.predicted_chi)"
            println(io, join([
                "true",
                r.n_qubits,
                r.seed,
                r.t,
                r.rank,
                r.nullity,
                chi_str,
                isnan(r.nullity_ratio) ? "" : @sprintf("%.6f", r.nullity_ratio)
            ], ","))
        end
    end
end

function write_model_csv(fit_results, ns, ranks, output_path::String)
    open(output_path, "w") do io
        println(io, "n_qubits,rank,model1_predicted,model2_predicted,model3_predicted")
        m1 = fit_results.model1
        m2 = fit_results.model2
        m3 = fit_results.model3
        for (n, r) in zip(ns, ranks)
            pred1 = m1.a * n^m1.b
            pred2 = m2.c * n * log(n)^m2.a
            pred3 = m3.c * n^m3.alpha * log(n)^m3.a
            @printf(io, "%d,%.6f,%.6f,%.6f,%.6f\n", Int(n), r, pred1, pred2, pred3)
        end
    end
end

function write_exponent_csv(exponents::Vector, output_path::String)
    open(output_path, "w") do io
        println(io, "n_low,n_high,alpha_eff")
        for e in exponents
            @printf(io, "%.0f,%.0f,%.6f\n", e.n_low, e.n_high, e.alpha_eff)
        end
    end
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("LARGE-N QFT RANK SCALING — ONLINE RREF")
    println("="^80)
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    verified = run_verification()

    if !verified
        println("ABORTING: Verification failed. Fix the code before running large N.")
        return nothing
    end

    if mode == "verify"
        println("Verification-only mode. Done.")
        return nothing
    end

    small_ns = [4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64]
    large_ns = mode == "full" ? [128, 256, 512] : [128, 256]

    all_ns = vcat(small_ns, large_ns)

    all_results = NamedTuple[]

    println("="^80)
    println("RUNNING COMPLETE DATASET")
    println("="^80)
    println()

    for n in all_ns
        seed = Int(hash(("qft_scaling", n, 1)) % UInt32)

        t_start = time()
        result = qft_gf2_rank_online(n; seed=seed)
        elapsed = time() - t_start

        push!(all_results, result)

        @printf("  N=%4d: t=%10d, rank=%6d, nullity=%10d, ν/t=%.6f  (%.1fs)\n",
                n, result.t, result.rank, result.nullity,
                isnan(result.nullity_ratio) ? 0.0 : result.nullity_ratio, elapsed)
    end

    ns_float = Float64.([r.n_qubits for r in all_results])
    ranks_float = Float64.([r.rank for r in all_results])

    println()
    fit_results = run_model_fitting(ns_float, ranks_float)

    println("="^80)
    println("EFFECTIVE EXPONENTS (local α between consecutive N)")
    println("="^80)
    println()

    exponents = compute_effective_exponents(ns_float, ranks_float)
    @printf("%10s %10s %12s\n", "N_low", "N_high", "α_eff")
    println("-"^35)
    for e in exponents
        @printf("%10.0f %10.0f %12.4f\n", e.n_low, e.n_high, e.alpha_eff)
    end
    println()

    println("="^80)
    println("SUMMARY")
    println("="^80)
    println()

    @printf("Complete N range: %d to %d (%d data points)\n",
            Int(minimum(ns_float)), Int(maximum(ns_float)), length(ns_float))

    rank_pattern_holds = all(r -> r.rank == r.n_qubits - 2, all_results)
    if rank_pattern_holds
        println("Pattern: rank = N - 2 holds for ALL tested N values!")
        println("This is an exact linear relationship, not a power law.")
        println("The rank grows as Θ(N) with offset -2.")
    else
        println("Pattern: rank = N - 2 does NOT hold for all N.")
        for r in all_results
            if r.rank != r.n_qubits - 2
                @printf("  N=%d: rank=%d (expected N-2=%d)\n", r.n_qubits, r.rank, r.n_qubits - 2)
            end
        end
    end

    println()

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)

    csv_results = joinpath(output_dir, "experiment_qft_large_n.csv")
    write_results_csv(all_results, csv_results)
    println("Results saved to: $csv_results")

    csv_models = joinpath(output_dir, "experiment_qft_large_n_models.csv")
    write_model_csv(fit_results, ns_float, ranks_float, csv_models)
    println("Model fits saved to: $csv_models")

    csv_exponents = joinpath(output_dir, "experiment_qft_large_n_exponents.csv")
    write_exponent_csv(exponents, csv_exponents)
    println("Exponents saved to: $csv_exponents")

    println("\n" * "="^80)
    println("LARGE-N QFT RANK SCALING — COMPLETE")
    println("="^80)

    return all_results, fit_results
end

results = main()
