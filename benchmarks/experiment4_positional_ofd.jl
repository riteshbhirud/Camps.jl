"""
Experiment 4: Per-T-Gate Positional OFD Analysis
=================================================

For each T-gate in a circuit, records:
- Position index (1st, 2nd, ..., t-th T-gate)
- Whether OFD succeeds or fails at that position
- Number of free qubits with X/Y available
- Pauli weight of the twisted Pauli
- Incremental GF(2) rank change (independent or dependent)

This produces per-position OFD success curves that reveal WHERE in a circuit
OFD transitions from working to failing. Key insight: for random circuits,
OFD fails gradually as free qubits deplete; for structured algorithms, OFD
fails from the very first T-gate (all twisted Paulis are pure-Z).

Usage:
    julia benchmarks/experiment4_positional_ofd.jl [mode]

Modes:
    "test"   - Quick validation (n=4,8, few families)
    "medium" - Standard (n=4..64, all 14 families)

Output:
    results/experiment4_positional_ofd.csv          (per-T-gate data)
    results/experiment4_positional_ofd_summary.csv  (per-family summary)
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
    analyze_per_t_gate(gates, t_positions, n_qubits; seed=nothing)

Walk through the circuit gate-by-gate, and for each T-gate record:
- position index (k-th T-gate)
- OFD success/fail
- number of free qubits with X/Y in twisted Pauli
- Pauli weight
- whether this T-gate is GF(2)-independent of previous ones
"""
function analyze_per_t_gate(gates::Vector, t_positions::Vector{Int},
                             n_qubits::Int; seed::Union{Integer,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    state = CAMPSState(n_qubits)
    initialize!(state)

    records = []
    twisted_paulis_so_far = PauliOperator[]
    t_idx = 0

    for (idx, gate) in enumerate(gates)
        if gate isa Tuple
            gate_type = gate[1]
            qubits = gate[2]

            if gate_type == :T
                t_idx += 1
                qubit = qubits[1]

                P_twisted = compute_twisted_pauli(state, :Z, qubit)
                xb = xbit(P_twisted)

                n_free_with_xy = 0
                for j in 1:n_qubits
                    if state.free_qubits[j] && xb[j]
                        n_free_with_xy += 1
                    end
                end

                is_pure_z = !any(xb)

                weight = 0
                for j in 1:n_qubits
                    x_j, z_j = P_twisted[j]
                    if x_j || z_j
                        weight += 1
                    end
                end

                if isempty(twisted_paulis_so_far)
                    is_independent = any(xb)
                    rank_before = 0
                    rank_after = is_independent ? 1 : 0
                else
                    M_before = build_gf2_matrix(twisted_paulis_so_far)
                    rank_before = gf2_rank(M_before)
                    push!(twisted_paulis_so_far, P_twisted)
                    M_after = build_gf2_matrix(twisted_paulis_so_far)
                    rank_after = gf2_rank(M_after)
                    pop!(twisted_paulis_so_far)
                    is_independent = rank_after > rank_before
                end

                push!(twisted_paulis_so_far, P_twisted)

                ofd_success = n_free_with_xy > 0

                push!(records, (
                    t_index = t_idx,
                    gate_position = idx,
                    ofd_success = ofd_success,
                    n_free_with_xy = n_free_with_xy,
                    n_free_total = count(state.free_qubits),
                    pauli_weight = weight,
                    is_pure_z = is_pure_z,
                    is_gf2_independent = is_independent,
                    rank_after = rank_after,
                    nullity_after = t_idx - rank_after,
                    fractional_position = 0.0
                ))

                if ofd_success
                    control = find_disentangling_qubit(P_twisted, state.free_qubits)
                    if control !== nothing
                        D_gates = build_disentangler_gates(P_twisted, control)
                        D_flat = flatten_gate_sequence(D_gates)
                        apply_inverse_gates!(state.clifford, D_flat)
                        state.free_qubits[control] = false
                    end
                end

            elseif gate_type == :H
                apply_gate!(state, CliffordGate([(:H, qubits[1])], [qubits[1]]))
            elseif gate_type == :CNOT
                apply_gate!(state, CliffordGate([(:CNOT, qubits[1], qubits[2])], [qubits[1], qubits[2]]))
            elseif gate_type == :X
                apply_gate!(state, CliffordGate([(:X, qubits[1])], [qubits[1]]))
            elseif gate_type == :Z
                apply_gate!(state, CliffordGate([(:Z, qubits[1])], [qubits[1]]))
            elseif gate_type == :S
                apply_gate!(state, CliffordGate([(:S, qubits[1])], [qubits[1]]))
            elseif gate_type == :Sdag
                apply_gate!(state, CliffordGate([(:Sdag, qubits[1])], [qubits[1]]))
            elseif gate_type == :random2q
                q1, q2 = qubits[1], qubits[2]
                cliff = random_clifford(2)
                sparse = SparseGate(cliff, [q1, q2])
                apply!(state.clifford, sparse)
            end

        elseif gate isa CliffordGate
            apply_clifford_gate_to_state!(state, gate)

        elseif gate isa RotationGate
            if !is_clifford_angle(gate.angle)
                t_idx += 1

                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                xb = xbit(P_twisted)

                n_free_with_xy = 0
                for j in 1:n_qubits
                    if state.free_qubits[j] && xb[j]
                        n_free_with_xy += 1
                    end
                end

                is_pure_z = !any(xb)

                weight = 0
                for j in 1:n_qubits
                    x_j, z_j = P_twisted[j]
                    if x_j || z_j
                        weight += 1
                    end
                end

                if isempty(twisted_paulis_so_far)
                    is_independent = any(xb)
                    rank_before = 0
                    rank_after = is_independent ? 1 : 0
                else
                    M_before = build_gf2_matrix(twisted_paulis_so_far)
                    rank_before = gf2_rank(M_before)
                    push!(twisted_paulis_so_far, P_twisted)
                    M_after = build_gf2_matrix(twisted_paulis_so_far)
                    rank_after = gf2_rank(M_after)
                    pop!(twisted_paulis_so_far)
                    is_independent = rank_after > rank_before
                end

                push!(twisted_paulis_so_far, P_twisted)

                ofd_success = n_free_with_xy > 0

                push!(records, (
                    t_index = t_idx,
                    gate_position = idx,
                    ofd_success = ofd_success,
                    n_free_with_xy = n_free_with_xy,
                    n_free_total = count(state.free_qubits),
                    pauli_weight = weight,
                    is_pure_z = is_pure_z,
                    is_gf2_independent = is_independent,
                    rank_after = rank_after,
                    nullity_after = t_idx - rank_after,
                    fractional_position = 0.0
                ))

                if ofd_success
                    control = find_disentangling_qubit(P_twisted, state.free_qubits)
                    if control !== nothing
                        D_gates = build_disentangler_gates(P_twisted, control)
                        D_flat = flatten_gate_sequence(D_gates)
                        apply_inverse_gates!(state.clifford, D_flat)
                        state.free_qubits[control] = false
                    end
                end
            else
                clifford_gate = rotation_to_clifford(gate)
                if clifford_gate !== nothing
                    apply_clifford_gate_to_state!(state, clifford_gate)
                end
            end
        end
    end

    n_t_total = length(records)
    for i in 1:n_t_total
        records[i] = merge(records[i], (fractional_position = n_t_total > 1 ? (i-1)/(n_t_total-1) : 0.0,))
    end

    return records
end

function generate_experiments_exp4(mode::String)
    experiments = []

    n_range = mode == "test" ? [4, 8] : [4, 8, 16, 32, 64]
    n_realizations = mode == "test" ? 1 : 3

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
                seed = Int(hash(("exp4", tag, n, real)) % UInt32)

                params = Dict{Symbol, Any}(
                    :n_qubits => n,
                    :n_t_gates => n,
                    :seed => seed
                )

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
                    family_name = family_name,
                    n_qubits = n,
                    generator = :phase1,
                    params = params
                ))
            end
        end
    end

    for n in n_range
        for real in 1:n_realizations
            seed = Int(hash(("exp4_QFT", n, real)) % UInt32)
            push!(experiments, (
                family_name = "Quantum Fourier Transform",
                n_qubits = n,
                generator = :qft_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :density => :medium, :seed => seed)
            ))
        end
    end

    grover_n_max = mode == "test" ? 8 : 16
    grover_n_range = filter(n -> n <= grover_n_max, n_range)
    for n in grover_n_range
        for real in 1:n_realizations
            seed = Int(hash(("exp4_Grover", n, real)) % UInt32)
            max_iter = n <= 12 ? nothing : min(50, ceil(Int, π/4 * sqrt(2^n)))
            push!(experiments, (
                family_name = "Grover Search",
                n_qubits = n,
                generator = :grover_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :density => :full, :seed => seed,
                                           :max_iterations => max_iter)
            ))
        end
    end

    for n in n_range
        for real in 1:n_realizations
            seed = Int(hash(("exp4_VQE", n, real)) % UInt32)
            push!(experiments, (
                family_name = "VQE Hardware-Efficient Ansatz",
                n_qubits = n,
                generator = :vqe_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :layers => 2, :seed => seed)
            ))
        end
    end

    qaoa_n_range = filter(n -> n % 2 == 0, n_range)
    for n in qaoa_n_range
        for real in 1:n_realizations
            seed = Int(hash(("exp4_QAOA", n, real)) % UInt32)
            push!(experiments, (
                family_name = "QAOA MaxCut (p=1, 3-regular)",
                n_qubits = n,
                generator = :phase1,
                params = Dict{Symbol,Any}(:n_qubits => n, :n_t_gates => n, :seed => seed)
            ))
        end
    end

    surface_n_range = filter(n -> n >= 8, n_range)
    for n in surface_n_range
        for real in 1:n_realizations
            seed = Int(hash(("exp4_Surface", n, real)) % UInt32)
            push!(experiments, (
                family_name = "Surface Code",
                n_qubits = n,
                generator = :surface_large,
                params = Dict{Symbol,Any}(:n_qubits => n, :n_t_gates => max(4, div(n, 2)), :seed => seed)
            ))
        end
    end

    return experiments
end

function run_exp4(experiment)
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
        records = analyze_per_t_gate(gates, t_positions, n_qubits_actual; seed=params[:seed])

        return (
            success = true,
            family = family_name,
            n_qubits = n_qubits_actual,
            seed = params[:seed],
            records = records
        )
    catch e
        return (
            success = false,
            family = family_name,
            n_qubits = get(params, :n_qubits, 0),
            seed = params[:seed],
            records = [],
            error_msg = sprint(showerror, e)
        )
    end
end

function write_detailed_csv(all_results, output_path)
    open(output_path, "w") do io
        println(io, "family,n_qubits,seed,t_index,n_t_total,fractional_position,ofd_success,n_free_with_xy,n_free_total,pauli_weight,is_pure_z,is_gf2_independent,rank_after,nullity_after")

        for result in all_results
            if !result.success
                continue
            end

            n_t_total = length(result.records)

            for rec in result.records
                println(io, join([
                    "\"$(result.family)\"",
                    result.n_qubits,
                    result.seed,
                    rec.t_index,
                    n_t_total,
                    @sprintf("%.6f", rec.fractional_position),
                    rec.ofd_success,
                    rec.n_free_with_xy,
                    rec.n_free_total,
                    rec.pauli_weight,
                    rec.is_pure_z,
                    rec.is_gf2_independent,
                    rec.rank_after,
                    rec.nullity_after
                ], ","))
            end
        end
    end
end

function write_summary_csv(all_results, output_path)
    open(output_path, "w") do io
        println(io, "family,n_qubits,seed,n_t_gates,n_ofd_success,n_ofd_fail,ofd_rate,n_pure_z,pure_z_rate,n_gf2_independent,gf2_independence_rate,first_fail_position,mean_pauli_weight")

        for result in all_results
            if !result.success || isempty(result.records)
                continue
            end

            records = result.records
            n_t = length(records)
            n_ofd_success = count(r -> r.ofd_success, records)
            n_ofd_fail = n_t - n_ofd_success
            ofd_rate = n_t > 0 ? n_ofd_success / n_t : NaN
            n_pure_z = count(r -> r.is_pure_z, records)
            pure_z_rate = n_t > 0 ? n_pure_z / n_t : NaN
            n_indep = count(r -> r.is_gf2_independent, records)
            indep_rate = n_t > 0 ? n_indep / n_t : NaN

            first_fail_idx = findfirst(r -> !r.ofd_success, records)
            first_fail_pos = first_fail_idx !== nothing ? records[first_fail_idx].fractional_position : 1.0

            mean_weight = mean([r.pauli_weight for r in records])

            println(io, join([
                "\"$(result.family)\"",
                result.n_qubits,
                result.seed,
                n_t,
                n_ofd_success,
                n_ofd_fail,
                @sprintf("%.6f", ofd_rate),
                n_pure_z,
                @sprintf("%.6f", pure_z_rate),
                n_indep,
                @sprintf("%.6f", indep_rate),
                @sprintf("%.6f", first_fail_pos),
                @sprintf("%.2f", mean_weight)
            ], ","))
        end
    end
end

function print_summary_exp4(all_results)
    successful = filter(r -> r.success && !isempty(r.records), all_results)
    family_names = sort(unique([r.family for r in successful]))

    println()
    println("="^130)
    println("EXPERIMENT 4: PER-T-GATE POSITIONAL OFD ANALYSIS — SUMMARY")
    println("="^130)
    @printf("%-40s %5s %5s %8s %8s %8s %10s %10s\n",
            "Family", "n", "T", "OFD%", "PureZ%", "Indep%", "1st Fail", "Wt(avg)")
    println("-"^130)

    for family in family_names
        fr = filter(r -> r.family == family, successful)
        n_sizes = sort(unique([r.n_qubits for r in fr]))

        for n in n_sizes
            nr = filter(r -> r.n_qubits == n, fr)
            if isempty(nr)
                continue
            end

            all_records = vcat([r.records for r in nr]...)
            n_t_avg = mean([length(r.records) for r in nr])
            ofd_pct = 100 * mean([r.ofd_success ? 1.0 : 0.0 for r in all_records])
            purez_pct = 100 * mean([r.is_pure_z ? 1.0 : 0.0 for r in all_records])
            indep_pct = 100 * mean([r.is_gf2_independent ? 1.0 : 0.0 for r in all_records])
            mean_wt = mean([r.pauli_weight for r in all_records])

            first_fails = Float64[]
            for r in nr
                ff = findfirst(rec -> !rec.ofd_success, r.records)
                push!(first_fails, ff !== nothing ? r.records[ff].fractional_position : 1.0)
            end
            mean_first_fail = mean(first_fails)

            @printf("%-40s %5d %5.0f %7.1f%% %7.1f%% %7.1f%% %10.3f %10.1f\n",
                    family, n, n_t_avg, ofd_pct, purez_pct, indep_pct, mean_first_fail, mean_wt)
        end
    end
    println("="^130)
end

function main_exp4()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("EXPERIMENT 4: PER-T-GATE POSITIONAL OFD ANALYSIS")
    println("="^80)
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    experiments = generate_experiments_exp4(mode)
    n_total = length(experiments)
    println("Total experiments: $n_total")
    println()

    println("Starting per-T-gate analysis (Clifford-only, no MPS)...")
    println("-"^80)

    start_time = time()
    results = []

    for (i, exp) in enumerate(experiments)
        t0 = time()
        result = run_exp4(exp)
        dt = time() - t0
        push!(results, result)

        n_records = length(result.records)

        if !result.success
            err_msg = hasproperty(result, :error_msg) ? result.error_msg : "unknown"
            @printf("[%4d/%4d] FAIL %-35s n=%3d (%.1fs) — %s\n",
                    i, n_total, exp.family_name, exp.n_qubits, dt,
                    first(err_msg, 60))
        elseif i % 10 == 0 || i == n_total || dt > 2.0
            n_ofd = count(r -> r.ofd_success, result.records)
            ofd_pct = n_records > 0 ? 100 * n_ofd / n_records : 0.0
            @printf("[%4d/%4d] %-35s n=%3d t=%5d OFD=%.0f%% (%.1fs)\n",
                    i, n_total, exp.family_name, result.n_qubits,
                    n_records, ofd_pct, dt)
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

    detail_csv = joinpath(output_dir, "experiment4_positional_ofd.csv")
    write_detailed_csv(results, detail_csv)
    println("\nDetailed results saved to: $detail_csv")

    summary_csv = joinpath(output_dir, "experiment4_positional_ofd_summary.csv")
    write_summary_csv(results, summary_csv)
    println("Summary saved to: $summary_csv")

    print_summary_exp4(results)

    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    results = main_exp4()
end
