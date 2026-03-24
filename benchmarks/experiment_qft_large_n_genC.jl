"""
Experiment: Large-N QFT Rank Scaling — Generator C (Exact Angles)
=================================================================

Uses qft_circuit() from simulation.jl which decomposes controlled-R_k gates
using EXACT rotation angles (2π/2^m), NOT the T-gate approximation.

This is the physically correct QFT decomposition. Each controlled-R_m (m≥2)
becomes:  Rz(angle/2), CNOT, Rz(-angle/2), CNOT, Rz(angle/2)
where angle = 2π / 2^m.

For m=2: angle = π/2 → Rz(π/4) which IS a T-gate (non-Clifford)
For m≥3: angle = π/2^(m-1) → Rz(π/2^m) which is NOT a multiple of π/4

Key difference from Generator A/B:
- Generator A/B: 4(k-2) T-gates per controlled-R_k for k≥4 → rank = N-2
- Generator C: 3 exact-angle rotations per controlled-R_m for m≥2 → rank = N-1

Uses the same online RREF approach as experiment_qft_large_n.jl.

Usage:
    julia benchmarks/experiment_qft_large_n_genC.jl [mode]

Modes:
    "verify"  - Verify at N=4,6,8,12,16 against known results
    "medium"  - Verify + run up to N=128, 256
    "full"    - Verify + run up to N=128, 256, 512

Output:
    results/experiment_qft_large_n_genC.csv
    results/experiment_qft_large_n_genC_models.csv
    results/experiment_qft_large_n_genC_exponents.csv
"""

camps_dir = dirname(dirname(@__FILE__))
println("Activating CAMPS.jl environment: $camps_dir")
using Pkg
Pkg.activate(camps_dir)

using CAMPS
using QuantumClifford
using Random
using Statistics
using Printf
using Dates

mutable struct OnlineRREF
    n::Int
    matrix::Matrix{Bool}
    pivot_col::Vector{Int}
    rank::Int
end

function OnlineRREF(n::Int)
    return OnlineRREF(n, zeros(Bool, n, n), zeros(Int, n), 0)
end

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
    qft_gf2_rank_genC(n_qubits::Int) -> NamedTuple

Compute GF(2) rank of QFT circuit using Generator C (qft_circuit from
simulation.jl) with online RREF.

Generator C uses exact rotation angles (2π/2^m) for controlled phase gates.
The circuit is deterministic (no seed needed).

Pipeline:
  1. Generate circuit via qft_circuit(n)
  2. Walk through gates with CAMPSState (Clifford-only, no MPS)
  3. At each non-Clifford RotationGate: compute twisted Pauli, extract x-bits,
     try_insert! into online RREF
  4. Apply OFD if possible (to match standard CAMPS behavior)
"""
function qft_gf2_rank_genC(n_qubits::Int)
    circuit = qft_circuit(n_qubits)

    state = CAMPSState(n_qubits)
    initialize!(state)

    rref = OnlineRREF(n_qubits)
    t_count = 0
    n_ofd = 0
    n_non_ofd = 0

    for gate in circuit
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
                    n_ofd += 1
                else
                    n_non_ofd += 1
                end
            end
        end
    end

    nullity = t_count - rref.rank
    predicted_chi = nullity >= 62 ? (typemax(Int) ÷ 2) : (nullity <= 0 ? 1 : 2^nullity)
    nullity_ratio = t_count > 0 ? nullity / t_count : NaN

    return (
        n_qubits = n_qubits,
        t = t_count,
        rank = rref.rank,
        nullity = nullity,
        predicted_chi = predicted_chi,
        nullity_ratio = nullity_ratio,
        n_ofd = n_ofd,
        n_non_ofd = n_non_ofd
    )
end

const KNOWN_GENC_RESULTS = Dict(
    4  => (t=18,  rank=3),
    6  => (t=45,  rank=5),
    8  => (t=84,  rank=7),
    12 => (t=198, rank=11),
    16 => (t=360, rank=15),
)

function run_verification()
    println("="^80)
    println("STEP 1: VERIFICATION — Generator C (exact angles)")
    println("="^80)
    println()

    all_pass = true
    n_range = sort(collect(keys(KNOWN_GENC_RESULTS)))

    @printf("%6s %10s %10s %10s %10s %8s\n",
            "N", "t_expect", "t_got", "rank_exp", "rank_got", "Status")
    println("-"^60)

    for n in n_range
        known = KNOWN_GENC_RESULTS[n]

        t_start = time()
        result = qft_gf2_rank_genC(n)
        elapsed = time() - t_start

        t_match = result.t == known.t
        rank_match = result.rank == known.rank
        pass = t_match && rank_match

        status = pass ? "PASS" : "FAIL"
        if !pass
            all_pass = false
        end

        expected_pattern = "N-1=$(n-1)"
        actual_pattern = result.rank == n - 1 ? "N-1" : (result.rank == n - 2 ? "N-2" : "other")

        @printf("%6d %10d %10d %10d %10d %8s  (%.2fs)  [%s]\n",
                n, known.t, result.t, known.rank, result.rank, status, elapsed, actual_pattern)
    end

    println("-"^60)
    if all_pass
        println("ALL VERIFICATIONS PASSED.")
    else
        println("VERIFICATION FAILED.")
    end
    println()

    return all_pass
end

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
    det_XtX = XtX[1,1] * (XtX[2,2]*XtX[3,3] - XtX[2,3]*XtX[3,2]) -
              XtX[1,2] * (XtX[2,1]*XtX[3,3] - XtX[2,3]*XtX[3,1]) +
              XtX[1,3] * (XtX[2,1]*XtX[3,2] - XtX[2,2]*XtX[3,1])

    abs(det_XtX) < 1e-12 && return (NaN, NaN, NaN, NaN, NaN)

    beta = XtX \ (X' * Y)

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

function compute_bic(n_points::Int, n_params::Int, rss::Float64)::Float64
    n_points <= 0 && return NaN
    rss <= 0 && return -Inf
    return n_points * log(rss / n_points) + n_params * log(n_points)
end

function run_model_fitting(ns::Vector{Float64}, ranks::Vector{Float64})
    println("="^80)
    println("MODEL FITTING AND SELECTION")
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
        println(io, "n_qubits,t,rank,nullity,predicted_chi,nullity_ratio,n_ofd,n_non_ofd,rank_eq_n_minus_1")
        for r in results
            chi_str = r.predicted_chi >= typemax(Int) ÷ 2 ? "overflow" : "$(r.predicted_chi)"
            println(io, join([
                r.n_qubits,
                r.t,
                r.rank,
                r.nullity,
                chi_str,
                isnan(r.nullity_ratio) ? "" : @sprintf("%.6f", r.nullity_ratio),
                r.n_ofd,
                r.n_non_ofd,
                r.rank == r.n_qubits - 1 ? "true" : "false"
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
    println("LARGE-N QFT RANK SCALING — GENERATOR C (EXACT ANGLES)")
    println("="^80)
    println("Generator: qft_circuit() from simulation.jl")
    println("Decomposition: Rz(angle/2), CNOT, Rz(-angle/2), CNOT, Rz(angle/2)")
    println("Angles: exact 2π/2^m (NOT T-gate approximation)")
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    verified = run_verification()

    if !verified
        println("ABORTING: Verification failed.")
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

    @printf("%6s %10s %8s %10s %10s %8s %8s %10s\n",
            "N", "t", "rank", "nullity", "ν/t", "OFD", "non-OFD", "time")
    println("-"^80)

    for n in all_ns
        t_start = time()
        result = qft_gf2_rank_genC(n)
        elapsed = time() - t_start

        push!(all_results, result)

        pattern = result.rank == n - 1 ? "N-1 ✓" : "N-1 ✗"

        @printf("%6d %10d %8d %10d %10.6f %8d %8d %8.1fs  [%s]\n",
                n, result.t, result.rank, result.nullity,
                isnan(result.nullity_ratio) ? 0.0 : result.nullity_ratio,
                result.n_ofd, result.n_non_ofd, elapsed, pattern)
    end
    println("-"^80)

    ns_float = Float64.([r.n_qubits for r in all_results])
    ranks_float = Float64.([r.rank for r in all_results])

    println()
    fit_results = run_model_fitting(ns_float, ranks_float)

    println("="^80)
    println("EFFECTIVE EXPONENTS")
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
    println("SUMMARY — GENERATOR C (EXACT ANGLES)")
    println("="^80)
    println()

    @printf("N range: %d to %d (%d data points)\n",
            Int(minimum(ns_float)), Int(maximum(ns_float)), length(ns_float))

    pattern_holds = all(r -> r.rank == r.n_qubits - 1, all_results)
    if pattern_holds
        println()
        println("RESULT: rank = N - 1 holds EXACTLY for ALL tested N.")
        println()
        println("Comparison with Generator A/B (T-gate approximation):")
        println("  Generator A/B: rank = N - 2  (one rank lower)")
        println("  Generator C:   rank = N - 1  (this experiment)")
        println()
        println("Both are exact linear relationships. The difference is due to")
        println("the decomposition: Generator C's exact angles produce one")
        println("additional independent GF(2) row compared to Generator A/B.")
    else
        println()
        println("RESULT: rank = N - 1 does NOT hold for all N.")
        for r in all_results
            if r.rank != r.n_qubits - 1
                @printf("  N=%d: rank=%d (expected N-1=%d)\n",
                        r.n_qubits, r.rank, r.n_qubits - 1)
            end
        end
    end

    println()

    println("T-COUNT COMPARISON (Generator C vs Generator A/B):")
    println("-"^60)
    @printf("%6s %12s %12s %10s\n", "N", "GenC t", "GenA/B t", "ratio")
    println("-"^60)
    gen_ab_t = Dict(4=>16, 6=>80, 8=>224, 10=>480, 12=>880,
                    16=>2240, 20=>4560, 24=>8096, 32=>19840,
                    48=>69184, 64=>166656)
    for r in all_results
        if haskey(gen_ab_t, r.n_qubits)
            t_ab = gen_ab_t[r.n_qubits]
            @printf("%6d %12d %12d %10.1fx\n",
                    r.n_qubits, r.t, t_ab, t_ab / r.t)
        else
            @printf("%6d %12d %12s %10s\n",
                    r.n_qubits, r.t, "—", "—")
        end
    end
    println("-"^60)

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)

    csv_results = joinpath(output_dir, "experiment_qft_large_n_genC.csv")
    write_results_csv(all_results, csv_results)
    println("\nResults saved to: $csv_results")

    csv_models = joinpath(output_dir, "experiment_qft_large_n_genC_models.csv")
    write_model_csv(fit_results, ns_float, ranks_float, csv_models)
    println("Model fits saved to: $csv_models")

    csv_exponents = joinpath(output_dir, "experiment_qft_large_n_genC_exponents.csv")
    write_exponent_csv(exponents, csv_exponents)
    println("Exponents saved to: $csv_exponents")

    println("\n" * "="^80)
    println("LARGE-N QFT RANK SCALING (GENERATOR C) — COMPLETE")
    println("="^80)

    return all_results, fit_results
end

results = main()
