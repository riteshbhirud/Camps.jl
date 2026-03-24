"""
Experiment 3: T-Gate Consolidation Preprocessing
=================================================

Explores whether consolidating adjacent T-gates before CAMPS simulation
can reduce T-count and improve efficiency. Two consolidation rules:

  Rule 1: T · T = S (Clifford) → eliminates 2 T-gates
  Rule 2: T · T† = I (identity) → eliminates 2 T-gates

After consolidation, re-run GF(2) analysis to see the effect on
nullity, rank, and predicted bond dimension.

This experiment quantifies the "low-hanging fruit" of circuit preprocessing:
many circuits (especially from gate synthesis) contain redundant T-gate pairs
that can be absorbed into the Clifford layer.

Usage:
    julia benchmarks/experiment3_t_consolidation.jl [mode]

Modes:
    "test"   - Quick validation (n=4,8)
    "medium" - Standard (n=4..64, all families)

Output:
    results/experiment3_t_consolidation.csv
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

"""
    consolidate_t_gates(gates::Vector, t_positions::Vector{Int})

Scan the circuit for adjacent T-gates on the same qubit and consolidate:
  - T·T → S (Clifford): replace two T-gates with one S gate
  - T·T† → I: remove both gates entirely
  - T†·T† → S† (Clifford): replace two T†-gates with one S† gate
  - T†·T → I: remove both

Returns: (new_gates, new_t_positions, consolidation_stats)

The consolidation only considers gates that are "adjacent" in the sense that
no other gate acts on the same qubit between them. Clifford gates on OTHER
qubits can be between them.
"""
function consolidate_t_gates(gates::Vector, t_positions::Vector{Int})
    n_gates = length(gates)

    action = fill(:keep, n_gates)

    n_tt_to_s = 0
    n_tdag_tdag_to_sdag = 0
    n_t_tdag_cancel = 0

    for i in 1:n_gates
        if action[i] != :keep
            continue
        end

        qubit_i, angle_i = get_t_gate_info(gates[i])
        if qubit_i === nothing
            continue
        end

        for j in (i+1):n_gates
            if action[j] != :keep
                continue
            end

            qubit_j = get_gate_qubit(gates[j])
            if qubit_j === nothing
                continue
            end

            if qubit_j != qubit_i
                continue
            end

            _, angle_j = get_t_gate_info(gates[j])
            if angle_j === nothing
                break
            end

            if isapprox(angle_i, π/4) && isapprox(angle_j, π/4)
                action[i] = :replace_with_s
                action[j] = :remove
                n_tt_to_s += 1
            elseif isapprox(angle_i, -π/4) && isapprox(angle_j, -π/4)
                action[i] = :replace_with_sdag
                action[j] = :remove
                n_tdag_tdag_to_sdag += 1
            elseif (isapprox(angle_i, π/4) && isapprox(angle_j, -π/4)) ||
                   (isapprox(angle_i, -π/4) && isapprox(angle_j, π/4))
                action[i] = :remove
                action[j] = :remove
                n_t_tdag_cancel += 1
            end
            break
        end
    end

    new_gates = []
    new_t_positions = Int[]

    for i in 1:n_gates
        if action[i] == :keep
            push!(new_gates, gates[i])
            q, a = get_t_gate_info(gates[i])
            if q !== nothing
                push!(new_t_positions, length(new_gates))
            end
        elseif action[i] == :replace_with_s
            qubit_i, _ = get_t_gate_info(gates[i])
            push!(new_gates, make_s_gate(gates[i], qubit_i))
        elseif action[i] == :replace_with_sdag
            qubit_i, _ = get_t_gate_info(gates[i])
            push!(new_gates, make_sdag_gate(gates[i], qubit_i))
        end
    end

    stats = (
        original_gates = n_gates,
        original_t_count = length(t_positions),
        consolidated_gates = length(new_gates),
        consolidated_t_count = length(new_t_positions),
        n_tt_to_s = n_tt_to_s,
        n_tdag_tdag_to_sdag = n_tdag_tdag_to_sdag,
        n_t_tdag_cancel = n_t_tdag_cancel,
        total_pairs_consolidated = n_tt_to_s + n_tdag_tdag_to_sdag + n_t_tdag_cancel,
        t_reduction = length(t_positions) - length(new_t_positions),
        t_reduction_pct = length(t_positions) > 0 ?
            (length(t_positions) - length(new_t_positions)) / length(t_positions) : 0.0
    )

    return (new_gates, new_t_positions, stats)
end

"""
Get (qubit, angle) if gate is a T-type gate, else (nothing, nothing).
"""
function get_t_gate_info(gate)
    if gate isa Tuple
        if gate[1] == :T
            return (gate[2][1], π/4)
        end
        return (nothing, nothing)
    elseif gate isa RotationGate
        if !is_clifford_angle(gate.angle)
            return (gate.qubit, gate.angle)
        end
        return (nothing, nothing)
    end
    return (nothing, nothing)
end

"""
Get the single qubit a gate acts on, or nothing for multi-qubit gates.
"""
function get_gate_qubit(gate)
    if gate isa Tuple
        qubits = gate[2]
        return length(qubits) == 1 ? qubits[1] : nothing
    elseif gate isa RotationGate
        return gate.qubit
    elseif gate isa CliffordGate
        return length(gate.qubits) == 1 ? gate.qubits[1] : nothing
    end
    return nothing
end

"""
Make an S gate in the same format as the original gate.
"""
function make_s_gate(original_gate, qubit::Int)
    if original_gate isa Tuple
        return (:S, [qubit])
    else
        return CliffordGate([(:S, qubit)], [qubit])
    end
end

"""
Make an S† gate in the same format as the original gate.
"""
function make_sdag_gate(original_gate, qubit::Int)
    if original_gate isa Tuple
        return (:Sdag, [qubit])
    else
        return CliffordGate([(:Sdag, qubit)], [qubit])
    end
end

function generate_experiments_exp3(mode::String)
    experiments = []

    n_range = mode == "test" ? [4, 8] : [4, 8, 16, 32, 64]
    n_realizations = mode == "test" ? 2 : 4

    phase1_families = [
        ("Random Clifford+T (Brick-wall)", :brick),
        ("Random Clifford+T (All-to-all)", :a2a),
        ("Bernstein-Vazirani", :bv),
        ("Simon's Algorithm", :simon),
        ("Deutsch-Jozsa", :dj),
        ("GHZ State", :ghz),
        ("Bell State / EPR Pairs", :bell),
        ("Graph State", :graph),
        ("Cluster State (1D)", :cluster),
    ]

    for (family_name, tag) in phase1_families
        for n in n_range
            if tag == :simon && n % 2 != 0
                continue
            end

            for real in 1:n_realizations
                seed = Int(hash(("exp3", tag, n, real)) % UInt32)
                params = Dict{Symbol, Any}(
                    :n_qubits => n, :n_t_gates => n, :seed => seed)

                if tag == :brick
                    params[:clifford_depth] = 2
                elseif tag == :a2a
                    params[:clifford_layers] = 2 * n
                elseif tag == :dj
                    rng_temp = Random.MersenneTwister(seed)
                    params[:function_type] = rand(rng_temp, [:constant, :balanced])
                elseif tag == :graph
                    params[:edge_probability] = 0.3
                end

                push!(experiments, (
                    family_name = family_name, n_qubits = n,
                    generator = :phase1, params = params))
            end
        end
    end

    for n in n_range
        for real in 1:n_realizations
            seed = Int(hash(("exp3_QFT", n, real)) % UInt32)
            push!(experiments, (
                family_name = "Quantum Fourier Transform", n_qubits = n,
                generator = :qft_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :density => :medium, :seed => seed)))
        end
    end

    grover_n_max = mode == "test" ? 8 : 16
    for n in filter(x -> x <= grover_n_max, n_range)
        for real in 1:n_realizations
            seed = Int(hash(("exp3_Grover", n, real)) % UInt32)
            max_iter = n <= 12 ? nothing : min(50, ceil(Int, π/4 * sqrt(2^n)))
            push!(experiments, (
                family_name = "Grover Search", n_qubits = n,
                generator = :grover_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :density => :full, :seed => seed,
                                           :max_iterations => max_iter)))
        end
    end

    for n in n_range
        for real in 1:n_realizations
            seed = Int(hash(("exp3_VQE", n, real)) % UInt32)
            push!(experiments, (
                family_name = "VQE Hardware-Efficient Ansatz", n_qubits = n,
                generator = :vqe_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :layers => 2, :seed => seed)))
        end
    end

    for n in filter(x -> x % 2 == 0, n_range)
        for real in 1:n_realizations
            seed = Int(hash(("exp3_QAOA", n, real)) % UInt32)
            push!(experiments, (
                family_name = "QAOA MaxCut (p=1, 3-regular)", n_qubits = n,
                generator = :phase1,
                params = Dict{Symbol,Any}(:n_qubits => n, :n_t_gates => n, :seed => seed)))
        end
    end

    for n in filter(x -> x >= 8, n_range)
        for real in 1:n_realizations
            seed = Int(hash(("exp3_Surface", n, real)) % UInt32)
            push!(experiments, (
                family_name = "Surface Code", n_qubits = n,
                generator = :surface_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :n_t_gates => max(4, div(n, 2)), :seed => seed)))
        end
    end

    return experiments
end

function run_exp3(experiment)
    family_name = experiment.family_name
    params = experiment.params
    gen = experiment.generator

    try
        Random.seed!(params[:seed])

        circuit_result = if gen == :qft_large
            generate_qft_circuit_large(params[:n_qubits];
                density=params[:density], seed=params[:seed])
        elseif gen == :grover_large
            generate_grover_circuit_large(params[:n_qubits];
                density=params[:density], seed=params[:seed],
                max_iterations=get(params, :max_iterations, nothing))
        elseif gen == :vqe_large
            generate_vqe_circuit_large(params[:n_qubits];
                layers=params[:layers], seed=params[:seed])
        elseif gen == :surface_large
            generate_surface_code_large(params[:n_qubits];
                n_t_gates=params[:n_t_gates], seed=params[:seed])
        elseif gen == :phase1
            family = get_family_from_name(family_name)
            generate_circuit(family, params)
        else
            error("Unknown generator: $gen")
        end

        if circuit_result isa CircuitInstance
            n_qubits_actual = circuit_result.n_qubits
            gates = circuit_result.gates
            t_positions = circuit_result.t_gate_positions
        else
            n_qubits_actual = circuit_result.n_qubits
            gates = circuit_result.gates
            t_positions = circuit_result.t_positions
        end

        Random.seed!(params[:seed])
        gf2_orig = compute_gf2_for_mixed_circuit(
            gates, t_positions, n_qubits_actual;
            seed=params[:seed], simulate_ofd=true)

        new_gates, new_t_positions, consol_stats = consolidate_t_gates(gates, t_positions)

        Random.seed!(params[:seed])
        gf2_consol = compute_gf2_for_mixed_circuit(
            Vector(new_gates), new_t_positions, n_qubits_actual;
            seed=params[:seed], simulate_ofd=true)

        return (
            success = true,
            family = family_name,
            n_qubits = n_qubits_actual,
            seed = params[:seed],
            orig_t_count = gf2_orig.n_t_gates,
            orig_total_gates = length(gates),
            orig_rank = gf2_orig.gf2_rank,
            orig_nullity = gf2_orig.nullity,
            orig_predicted_chi = gf2_orig.predicted_chi,
            orig_ofd_rate = gf2_orig.n_t_gates > 0 ?
                gf2_orig.n_disentanglable / gf2_orig.n_t_gates : NaN,
            consol_t_count = gf2_consol.n_t_gates,
            consol_total_gates = length(new_gates),
            consol_rank = gf2_consol.gf2_rank,
            consol_nullity = gf2_consol.nullity,
            consol_predicted_chi = gf2_consol.predicted_chi,
            consol_ofd_rate = gf2_consol.n_t_gates > 0 ?
                gf2_consol.n_disentanglable / gf2_consol.n_t_gates : NaN,
            n_tt_to_s = consol_stats.n_tt_to_s,
            n_tdag_tdag_to_sdag = consol_stats.n_tdag_tdag_to_sdag,
            n_t_tdag_cancel = consol_stats.n_t_tdag_cancel,
            t_reduction = consol_stats.t_reduction,
            t_reduction_pct = consol_stats.t_reduction_pct,
            nullity_reduction = gf2_orig.nullity - gf2_consol.nullity,
            chi_improvement = gf2_orig.predicted_chi > 0 ?
                gf2_consol.predicted_chi / gf2_orig.predicted_chi : NaN
        )
    catch e
        return (
            success = false,
            family = family_name,
            n_qubits = get(params, :n_qubits, 0),
            seed = params[:seed],
            orig_t_count = 0, orig_total_gates = 0,
            orig_rank = 0, orig_nullity = 0, orig_predicted_chi = 0,
            orig_ofd_rate = NaN,
            consol_t_count = 0, consol_total_gates = 0,
            consol_rank = 0, consol_nullity = 0, consol_predicted_chi = 0,
            consol_ofd_rate = NaN,
            n_tt_to_s = 0, n_tdag_tdag_to_sdag = 0, n_t_tdag_cancel = 0,
            t_reduction = 0, t_reduction_pct = 0.0,
            nullity_reduction = 0, chi_improvement = NaN,
            error_msg = sprint(showerror, e)
        )
    end
end

function write_results_csv_exp3(results, output_path)
    open(output_path, "w") do io
        println(io, "success,family,n_qubits,seed,orig_t_count,orig_total_gates,orig_rank,orig_nullity,orig_predicted_chi,orig_ofd_rate,consol_t_count,consol_total_gates,consol_rank,consol_nullity,consol_predicted_chi,consol_ofd_rate,n_tt_to_s,n_tdag_tdag_to_sdag,n_t_tdag_cancel,t_reduction,t_reduction_pct,nullity_reduction,chi_improvement")

        for r in results
            println(io, join([
                r.success,
                "\"$(r.family)\"",
                r.n_qubits,
                r.seed,
                r.orig_t_count,
                r.orig_total_gates,
                r.orig_rank,
                r.orig_nullity,
                r.orig_predicted_chi,
                isnan(r.orig_ofd_rate) ? "" : @sprintf("%.6f", r.orig_ofd_rate),
                r.consol_t_count,
                r.consol_total_gates,
                r.consol_rank,
                r.consol_nullity,
                r.consol_predicted_chi,
                isnan(r.consol_ofd_rate) ? "" : @sprintf("%.6f", r.consol_ofd_rate),
                r.n_tt_to_s,
                r.n_tdag_tdag_to_sdag,
                r.n_t_tdag_cancel,
                r.t_reduction,
                @sprintf("%.6f", r.t_reduction_pct),
                r.nullity_reduction,
                isnan(r.chi_improvement) ? "" : @sprintf("%.6f", r.chi_improvement)
            ], ","))
        end
    end
end

function print_summary_exp3(results)
    successful = filter(r -> r.success, results)
    family_names = sort(unique([r.family for r in successful]))

    println()
    println("="^140)
    println("EXPERIMENT 3: T-GATE CONSOLIDATION PREPROCESSING — SUMMARY")
    println("="^140)
    @printf("%-40s %5s %6s %6s %6s %8s %8s %8s %8s\n",
            "Family", "n", "T_orig", "T_new", "ΔT", "ν_orig", "ν_new", "χ_orig", "χ_new")
    println("-"^140)

    for family in family_names
        fr = filter(r -> r.family == family, successful)
        n_sizes = sort(unique([r.n_qubits for r in fr]))

        for n in n_sizes
            nr = filter(r -> r.n_qubits == n, fr)
            if isempty(nr)
                continue
            end

            m_orig_t = mean([r.orig_t_count for r in nr])
            m_consol_t = mean([r.consol_t_count for r in nr])
            m_dt = mean([r.t_reduction for r in nr])
            m_orig_null = mean([r.orig_nullity for r in nr])
            m_consol_null = mean([r.consol_nullity for r in nr])
            m_orig_chi = mean([r.orig_predicted_chi for r in nr])
            m_consol_chi = mean([r.consol_predicted_chi for r in nr])

            @printf("%-40s %5d %6.0f %6.0f %6.0f %8.1f %8.1f %8.0f %8.0f\n",
                    family, n, m_orig_t, m_consol_t, m_dt,
                    m_orig_null, m_consol_null, m_orig_chi, m_consol_chi)
        end
    end
    println("="^140)

    total_orig_t = sum(r.orig_t_count for r in successful)
    total_consol_t = sum(r.consol_t_count for r in successful)
    println()
    println("Overall: $(total_orig_t) → $(total_consol_t) T-gates " *
            "($(total_orig_t - total_consol_t) eliminated, " *
            "$(round(100*(total_orig_t - total_consol_t)/max(1,total_orig_t), digits=1))%)")
end

function main_exp3()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("EXPERIMENT 3: T-GATE CONSOLIDATION PREPROCESSING")
    println("="^80)
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    experiments = generate_experiments_exp3(mode)
    n_total = length(experiments)
    println("Total experiments: $n_total")
    println()

    println("Starting T-gate consolidation analysis...")
    println("-"^80)

    start_time = time()
    results = []

    for (i, exp) in enumerate(experiments)
        t0 = time()
        result = run_exp3(exp)
        dt = time() - t0
        push!(results, result)

        if !result.success
            err_msg = hasproperty(result, :error_msg) ? result.error_msg : "unknown"
            @printf("[%4d/%4d] FAIL %-35s n=%3d (%.1fs) — %s\n",
                    i, n_total, exp.family_name, exp.n_qubits, dt,
                    first(err_msg, 60))
        elseif i % 10 == 0 || i == n_total || dt > 2.0
            @printf("[%4d/%4d] %-35s n=%3d T: %d→%d (Δ%d) ν: %d→%d (%.1fs)\n",
                    i, n_total, exp.family_name, result.n_qubits,
                    result.orig_t_count, result.consol_t_count, result.t_reduction,
                    result.orig_nullity, result.consol_nullity, dt)
        end
    end

    total_time = time() - start_time
    println("-"^80)
    @printf("Completed %d experiments in %.1f seconds\n", n_total, total_time)

    n_success = count(r -> r.success, results)
    n_failed = n_total - n_success
    println("Success: $n_success, Failed: $n_failed")

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)

    output_csv = joinpath(output_dir, "experiment3_t_consolidation.csv")
    write_results_csv_exp3(results, output_csv)
    println("\nResults saved to: $output_csv")

    print_summary_exp3(results)

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main_exp3()
end
