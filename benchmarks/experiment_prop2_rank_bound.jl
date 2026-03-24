"""
Experiment: Proposition 2 — Repeated-Block Rank Bound Verification
===================================================================

Validates Proposition 2 from Liu & Clark: for a circuit composed of r
repetitions of a sub-block containing g T-gates on N qubits, the GF(2)
rank of the twisted Pauli x-bit matrix satisfies:

    rank(z) ≤ min(r·g, g + (r−1)·N)

This is a pure GF(2) analysis — no MPS simulation needed. Uses:
  - VQE Hardware-Efficient Ansatz: layers parameter = repetition count r
  - Grover Search: max_iterations parameter = repetition count r

For each (family, n, r), the circuit is generated, GF(2) rank computed,
and verify the bound holds.

Usage:
    julia benchmarks/experiment_prop2_rank_bound.jl [mode]

Modes:
    "test"   - Quick validation (small n, few r values)
    "medium" - Standard sweep

Output:
    results/experiment_prop2_rank_bound.csv
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

include(joinpath(camps_dir, "benchmarks", "circuit_families_complete.jl"))
include(joinpath(camps_dir, "benchmarks", "gf2_scaling_analysis.jl"))

function generate_experiments(mode::String)
    experiments = []

    n_realizations = mode == "test" ? 2 : 4

    vqe_n_range = mode == "test" ? [4, 6] : [4, 6, 8, 10, 12]
    vqe_r_range = mode == "test" ? [1, 2, 5] : [1, 2, 3, 5, 10, 15, 20, 30, 50]

    for n in vqe_n_range
        for r in vqe_r_range
            for real in 1:n_realizations
                seed = Int(hash(("prop2_vqe", n, r, real)) % UInt32)
                push!(experiments, (
                    family = "VQE",
                    n_qubits = n,
                    repetitions = r,
                    seed = seed,
                    generator = :vqe
                ))
            end
        end
    end

    grover_n_range = mode == "test" ? [4, 6] : [4, 6, 8, 10, 12]
    grover_r_range = mode == "test" ? [1, 2, 5] : [1, 2, 3, 5, 10]

    for n in grover_n_range
        for r in grover_r_range
            for real in 1:n_realizations
                seed = Int(hash(("prop2_grover", n, r, real)) % UInt32)
                push!(experiments, (
                    family = "Grover",
                    n_qubits = n,
                    repetitions = r,
                    seed = seed,
                    generator = :grover
                ))
            end
        end
    end

    return experiments
end

function run_experiment(exp)
    try
        circuit = if exp.generator == :vqe
            generate_vqe_circuit_large(exp.n_qubits; layers=exp.repetitions, seed=exp.seed)
        else
            generate_grover_circuit_large(exp.n_qubits; density=:full, seed=exp.seed,
                                          max_iterations=exp.repetitions)
        end

        gates = circuit.gates
        t_positions = circuit.t_positions
        n_qubits = circuit.n_qubits

        Random.seed!(exp.seed)
        gf2 = compute_gf2_for_mixed_circuit(
            gates, t_positions, n_qubits;
            seed=exp.seed, simulate_ofd=true)

        t = gf2.n_t_gates
        rank = gf2.gf2_rank
        nullity = gf2.nullity

        ref_circuit = if exp.generator == :vqe
            generate_vqe_circuit_large(exp.n_qubits; layers=1, seed=exp.seed)
        else
            generate_grover_circuit_large(exp.n_qubits; density=:full, seed=exp.seed,
                                          max_iterations=1)
        end
        g = length(ref_circuit.t_positions)

        r = exp.repetitions
        N = n_qubits

        bound_rg = r * g
        bound_prop2 = g + (r - 1) * N
        bound = min(bound_rg, bound_prop2)

        bound_satisfied = rank <= bound

        return (
            success = true,
            family = exp.family,
            n_qubits = N,
            r = r,
            seed = exp.seed,
            t = t,
            g = g,
            rank = rank,
            nullity = nullity,
            bound_rg = bound_rg,
            bound_prop2 = bound_prop2,
            bound = bound,
            bound_satisfied = bound_satisfied,
            rank_to_bound_ratio = bound > 0 ? rank / bound : NaN
        )
    catch e
        return (
            success = false,
            family = exp.family,
            n_qubits = exp.n_qubits,
            r = exp.repetitions,
            seed = exp.seed,
            t = 0,
            g = 0,
            rank = 0,
            nullity = 0,
            bound_rg = 0,
            bound_prop2 = 0,
            bound = 0,
            bound_satisfied = false,
            rank_to_bound_ratio = NaN,
            error_msg = sprint(showerror, e)
        )
    end
end

function write_results_csv(results, output_path)
    open(output_path, "w") do io
        println(io, "success,family,n_qubits,r,seed,t,g,rank,nullity,bound_rg,bound_prop2,bound,bound_satisfied,rank_to_bound_ratio")

        for res in results
            println(io, join([
                res.success,
                "\"$(res.family)\"",
                res.n_qubits,
                res.r,
                res.seed,
                res.t,
                res.g,
                res.rank,
                res.nullity,
                res.bound_rg,
                res.bound_prop2,
                res.bound,
                res.bound_satisfied,
                isnan(res.rank_to_bound_ratio) ? "" : @sprintf("%.6f", res.rank_to_bound_ratio)
            ], ","))
        end
    end
end

function print_summary(results)
    successful = filter(r -> r.success, results)

    println()
    println("="^110)
    println("PROPOSITION 2 RANK BOUND VERIFICATION — SUMMARY")
    println("="^110)
    @printf("%-10s %4s %4s %6s %6s %6s %8s %8s %10s %8s\n",
            "Family", "N", "r", "t", "g", "rank", "bound", "rg·bound", "rank/bound", "Holds?")
    println("-"^110)

    for family in sort(unique([r.family for r in successful]))
        fr = filter(r -> r.family == family, successful)
        n_sizes = sort(unique([r.n_qubits for r in fr]))

        for n in n_sizes
            nr = filter(r -> r.n_qubits == n, fr)
            r_vals = sort(unique([r.r for r in nr]))

            for r_val in r_vals
                rr = filter(r -> r.r == r_val, nr)
                mean_t = mean([r.t for r in rr])
                mean_g = mean([r.g for r in rr])
                mean_rank = mean([r.rank for r in rr])
                mean_bound = mean([r.bound for r in rr])
                mean_bound_rg = mean([r.bound_rg for r in rr])
                ratios = filter(!isnan, [r.rank_to_bound_ratio for r in rr])
                mean_ratio = isempty(ratios) ? NaN : mean(ratios)
                all_satisfied = all(r -> r.bound_satisfied, rr)

                @printf("%-10s %4d %4d %6.0f %6.0f %6.0f %8.0f %8.0f %10.4f %8s\n",
                        family, n, r_val, mean_t, mean_g, mean_rank,
                        mean_bound, mean_bound_rg,
                        isnan(mean_ratio) ? 0.0 : mean_ratio,
                        all_satisfied ? "YES" : "NO")
            end
        end
    end

    n_total = length(successful)
    n_violations = count(r -> !r.bound_satisfied, successful)
    println("="^110)
    println("Total experiments: $n_total")
    println("Bound violations: $n_violations")
    if n_violations == 0
        println("Proposition 2 bound holds universally!")
    else
        println("\nViolation details:")
        for r in filter(r -> !r.bound_satisfied, successful)
            @printf("  %s N=%d r=%d: rank=%d > bound=%d\n",
                    r.family, r.n_qubits, r.r, r.rank, r.bound)
        end
    end
    println("="^110)
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("PROPOSITION 2: REPEATED-BLOCK RANK BOUND VERIFICATION")
    println("="^80)
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    experiments = generate_experiments(mode)
    n_total = length(experiments)
    println("Total experiments: $n_total")
    println()

    println("Starting experiments...")
    println("-"^80)

    start_time = time()
    results = []

    for (i, exp) in enumerate(experiments)
        t0 = time()
        result = run_experiment(exp)
        dt = time() - t0
        push!(results, result)

        if !result.success
            err_msg = hasproperty(result, :error_msg) ? result.error_msg : "unknown"
            @printf("[%4d/%4d] FAIL %-10s N=%2d r=%3d (%.1fs) — %s\n",
                    i, n_total, exp.family, exp.n_qubits, exp.repetitions, dt,
                    first(err_msg, 60))
        elseif i % 10 == 0 || i == n_total || dt > 2.0
            @printf("[%4d/%4d] %-10s N=%2d r=%3d t=%5d rank=%4d bound=%4d holds=%s (%.1fs)\n",
                    i, n_total, result.family, result.n_qubits, result.r,
                    result.t, result.rank, result.bound,
                    result.bound_satisfied ? "Y" : "N", dt)
        end
    end

    total_time = time() - start_time
    println("-"^80)
    @printf("Completed %d experiments in %.1f seconds\n", n_total, total_time)

    n_success = count(r -> r.success, results)
    n_failed = n_total - n_success
    println("Success: $n_success, Failed: $n_failed")

    if n_failed > 0
        println("\nFailed experiments:")
        for r in filter(r -> !r.success, results)
            msg = hasproperty(r, :error_msg) ? r.error_msg : "unknown error"
            println("  $(r.family) N=$(r.n_qubits) r=$(r.r): $(first(msg, 80))")
        end
    end

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)

    output_csv = joinpath(output_dir, "experiment_prop2_rank_bound.csv")
    write_results_csv(results, output_csv)
    println("\nResults saved to: $output_csv")

    print_summary(results)

    return results
end

results = main()
