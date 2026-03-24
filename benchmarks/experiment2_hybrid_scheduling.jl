"""
Experiment 2: GF(2)-Screened OFD — Pre-Filtering Validation
============================================================

Validates whether GF(2) pre-analysis can correctly identify OFD-amenable
T-gates, enabling a "screened" strategy that skips futile OFD attempts.

For each circuit, two strategies are compared (both use OFD + naive absorption,
NO OBD):
  (A) Default: attempt OFD on every T-gate; fall back to naive absorption
      when OFD fails.
  (B) GF(2)-Guided: pre-analyze with GF(2) to identify amenable T-gates.
      Only attempt OFD on those; immediately absorb the rest (skip the
      OFD attempt entirely).

If χ_guided ≈ χ_default, GF(2) is a perfect predictor of OFD success.
If χ_guided > χ_default, GF(2) misclassifies some gates.

Metrics: bond dimension, OFD attempts saved, time savings, GF(2) accuracy.

NOTE: This experiment works at small scale (n ≤ 8) because full simulation
with MPS is needed to measure actual bond dimensions.

Usage:
    julia benchmarks/experiment2_hybrid_scheduling.jl [mode]

Modes:
    "test"   - Quick validation (n=4,5 only)
    "medium" - Standard (n=4..8, all families)

Output:
    results/experiment2_hybrid_scheduling.csv
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

function get_family_from_name(family_name::String)
    if family_name == "Random Clifford+T (Brick-wall)"
        return RandomBrickwallCliffordT()
    elseif family_name == "Random Clifford+T (All-to-all)"
        return RandomAllToAllCliffordT()
    elseif family_name == "Bernstein-Vazirani"
        return BernsteinVaziraniCircuit()
    elseif family_name == "Simon's Algorithm"
        return SimonCircuit()
    elseif family_name == "Deutsch-Jozsa"
        return DeutschJozsaCircuit()
    elseif family_name == "GHZ State"
        return GHZStateCircuit()
    elseif family_name == "Bell State / EPR Pairs"
        return BellStateCircuit()
    elseif family_name == "Graph State"
        return GraphStateCircuit()
    elseif family_name == "Cluster State (1D)"
        return ClusterStateCircuit()
    elseif family_name == "QAOA MaxCut (p=1, 3-regular)"
        return QAOAMaxCutCircuit()
    elseif family_name == "Surface Code"
        return SurfaceCodeFamily()
    elseif family_name == "Quantum Fourier Transform"
        return QFTFamily()
    elseif family_name == "Grover Search"
        return GroverFamily()
    elseif family_name == "VQE Hardware-Efficient Ansatz"
        return VQEFamily()
    else
        error("Unknown family: $family_name")
    end
end

"""
Dispatch generate_circuit correctly for Phase 1 (Dict) vs Phase 2 (keyword) families.
"""
function generate_circuit_dispatch(family, params::Dict)
    if family isa QFTFamily
        return generate_circuit(family; n_qubits=params[:n_qubits],
                               density=params[:density], seed=params[:seed])
    elseif family isa GroverFamily
        return generate_circuit(family; n_qubits=params[:n_qubits],
                               density=params[:density], seed=params[:seed])
    elseif family isa VQEFamily
        return generate_circuit(family; n_qubits=params[:n_qubits],
                               layers=params[:layers], seed=params[:seed])
    elseif family isa SurfaceCodeFamily
        return generate_circuit(family; n_qubits=params[:n_qubits],
                               n_t_gates=params[:n_t_gates], seed=params[:seed])
    else
        return generate_circuit(family, params)
    end
end

"""
    analyze_t_gate_ofd_amenability(gates, t_positions, n_qubits; seed=nothing)

For each T-gate in the circuit, determine whether it would be OFD-amenable
by running a Clifford-only walk and checking xbit conditions.

Returns a vector of (t_gate_index, position_in_circuit, is_ofd_amenable) tuples.
"""
function analyze_t_gate_ofd_amenability(gates::Vector, t_positions::Vector{Int},
                                         n_qubits::Int; seed::Union{Integer,Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    state = CAMPSState(n_qubits)
    initialize!(state)

    amenability = Vector{NamedTuple{(:t_index, :gate_pos, :is_amenable, :n_free_with_xy),
                                     Tuple{Int, Int, Bool, Int}}}()

    t_idx = 0

    for (idx, gate) in enumerate(gates)
        is_t_gate = false

        if gate isa Tuple
            gate_type = gate[1]
            qubits = gate[2]

            if gate_type == :T
                is_t_gate = true
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

                is_amenable = n_free_with_xy > 0

                push!(amenability, (t_index=t_idx, gate_pos=idx,
                                     is_amenable=is_amenable, n_free_with_xy=n_free_with_xy))

                if is_amenable
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
                is_t_gate = true
                t_idx += 1

                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                xb = xbit(P_twisted)

                n_free_with_xy = 0
                for j in 1:n_qubits
                    if state.free_qubits[j] && xb[j]
                        n_free_with_xy += 1
                    end
                end

                is_amenable = n_free_with_xy > 0

                push!(amenability, (t_index=t_idx, gate_pos=idx,
                                     is_amenable=is_amenable, n_free_with_xy=n_free_with_xy))

                if is_amenable
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

    return amenability
end

"""
    simulate_with_gf2_screening(gates, t_positions, n_qubits;
                                 seed=nothing, amenable_indices=nothing)

Simulate a circuit with per-T-gate strategy dispatch.

If `amenable_indices` is nothing, attempt OFD on every T-gate (= default behavior).
If `amenable_indices` is a Set{Int}, attempt OFD only on T-gates whose sequential
index is in the set; all others are naively absorbed (NoDisentangling).

Returns: (final_chi, n_ofd_attempted, n_ofd_skipped, sim_time, ofd_rate)
"""
function simulate_with_gf2_screening(gates::Vector, t_positions::Vector{Int},
                                      n_qubits::Int;
                                      seed::Union{Integer,Nothing}=nothing,
                                      amenable_indices::Union{Set{Int},Nothing}=nothing)
    if seed !== nothing
        Random.seed!(seed)
    end

    state = CAMPSState(n_qubits; max_bond=512, cutoff=1e-14)
    initialize!(state)

    n_ofd_attempted = 0
    n_ofd_skipped = 0
    n_clifford = 0
    n_non_clifford = 0
    t_gate_counter = 0

    t0 = time()

    for gate in gates
        if gate isa Tuple
            gate_type = gate[1]
            qubits = gate[2]

            if gate_type == :T
                t_gate_counter += 1
                n_non_clifford += 1
                qubit = qubits[1]

                if amenable_indices === nothing || t_gate_counter in amenable_indices
                    n_ofd_attempted += 1
                    apply_gate!(state, TGate(qubit); strategy=OFDStrategy())
                else
                    n_ofd_skipped += 1
                    apply_gate!(state, TGate(qubit); strategy=NoDisentangling())
                end

            elseif gate_type == :H
                apply_gate!(state, CliffordGate([(:H, qubits[1])], [qubits[1]]); strategy=OFDStrategy())
                n_clifford += 1
            elseif gate_type == :CNOT
                apply_gate!(state, CliffordGate([(:CNOT, qubits[1], qubits[2])], [qubits[1], qubits[2]]); strategy=OFDStrategy())
                n_clifford += 1
            elseif gate_type == :X
                apply_gate!(state, CliffordGate([(:X, qubits[1])], [qubits[1]]); strategy=OFDStrategy())
                n_clifford += 1
            elseif gate_type == :Z
                apply_gate!(state, CliffordGate([(:Z, qubits[1])], [qubits[1]]); strategy=OFDStrategy())
                n_clifford += 1
            elseif gate_type == :S
                apply_gate!(state, CliffordGate([(:S, qubits[1])], [qubits[1]]); strategy=OFDStrategy())
                n_clifford += 1
            elseif gate_type == :Sdag
                apply_gate!(state, CliffordGate([(:Sdag, qubits[1])], [qubits[1]]); strategy=OFDStrategy())
                n_clifford += 1
            elseif gate_type == :random2q
                q1, q2 = qubits[1], qubits[2]
                cliff = random_clifford(2)
                sparse = SparseGate(cliff, [q1, q2])
                apply!(state.clifford, sparse)
                n_clifford += 1
            end

        elseif gate isa CliffordGate
            apply_gate!(state, gate; strategy=OFDStrategy())
            n_clifford += 1

        elseif gate isa RotationGate
            if is_clifford_angle(gate.angle)
                apply_gate!(state, gate; strategy=OFDStrategy())
                n_clifford += 1
            else
                t_gate_counter += 1
                n_non_clifford += 1

                if amenable_indices === nothing || t_gate_counter in amenable_indices
                    n_ofd_attempted += 1
                    apply_gate!(state, gate; strategy=OFDStrategy())
                else
                    n_ofd_skipped += 1
                    apply_gate!(state, gate; strategy=NoDisentangling())
                end
            end
        end
    end

    sim_time = time() - t0
    final_chi = get_bond_dimension(state)

    return (
        final_chi = final_chi,
        n_ofd_attempted = n_ofd_attempted,
        n_ofd_skipped = n_ofd_skipped,
        n_clifford = n_clifford,
        n_non_clifford = n_non_clifford,
        sim_time = sim_time,
        ofd_rate = n_non_clifford > 0 ? n_ofd_attempted / n_non_clifford : NaN
    )
end

function generate_experiments(mode::String)
    experiments = []

    n_range = mode == "test" ? [4, 5] : [4, 5, 6, 7, 8]
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
                seed = Int(hash(("exp2", tag, n, real)) % UInt32)

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
                    params = params,
                    phase = 1
                ))
            end
        end
    end

    phase2_n_range = mode == "test" ? [4, 5] : [4, 5, 6, 7, 8]

    qaoa_n_range = filter(n -> n % 2 == 0, phase2_n_range)
    for n in qaoa_n_range
        for real in 1:n_realizations
            seed = Int(hash(("exp2_QAOA", n, real)) % UInt32)
            push!(experiments, (
                family_name = "QAOA MaxCut (p=1, 3-regular)",
                n_qubits = n,
                params = Dict{Symbol,Any}(:n_qubits => n, :n_t_gates => n, :seed => seed),
                phase = 2
            ))
        end
    end

    for fam_name in ["Quantum Fourier Transform", "Grover Search",
                      "VQE Hardware-Efficient Ansatz", "Surface Code"]
        for n in phase2_n_range
            for real in 1:n_realizations
                seed = Int(hash(("exp2_$fam_name", n, real)) % UInt32)

                params = Dict{Symbol,Any}(:n_qubits => n, :seed => seed)
                if fam_name == "Surface Code"
                    params[:n_t_gates] = max(4, div(n, 2))
                elseif fam_name == "Quantum Fourier Transform"
                    params[:density] = :medium
                elseif fam_name == "Grover Search"
                    params[:density] = :half
                elseif fam_name == "VQE Hardware-Efficient Ansatz"
                    params[:layers] = 2
                end

                push!(experiments, (
                    family_name = fam_name,
                    n_qubits = n,
                    params = params,
                    phase = 2
                ))
            end
        end
    end

    return experiments
end

function run_experiment(experiment)
    family_name = experiment.family_name
    params = experiment.params
    n = experiment.n_qubits

    try
        Random.seed!(params[:seed])

        family = get_family_from_name(family_name)
        circuit_result = generate_circuit_dispatch(family, params)

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
        family2 = get_family_from_name(family_name)
        generate_circuit_dispatch(family2, params)
        Random.seed!(params[:seed])
        family3 = get_family_from_name(family_name)
        circuit_result2 = generate_circuit_dispatch(family3, params)

        if circuit_result2 isa CircuitInstance
            gates2 = circuit_result2.gates
            t_positions2 = circuit_result2.t_gate_positions
            n_q2 = circuit_result2.n_qubits
        else
            gates2 = circuit_result2.gates
            t_positions2 = circuit_result2.t_positions
            n_q2 = circuit_result2.n_qubits
        end

        amenability = analyze_t_gate_ofd_amenability(gates2, t_positions2, n_q2; seed=params[:seed])

        n_t_total = length(amenability)
        n_amenable = count(a -> a.is_amenable, amenability)
        n_not_amenable = n_t_total - n_amenable
        amenability_rate = n_t_total > 0 ? n_amenable / n_t_total : NaN

        Random.seed!(params[:seed])
        gf2_result = compute_gf2_for_mixed_circuit(
            gates2, t_positions2, n_q2; seed=params[:seed], simulate_ofd=true)

        gf2_rank = gf2_result.gf2_rank
        nullity = gf2_result.nullity
        nullity_ratio = n_t_total > 0 ? nullity / n_t_total : NaN

        Random.seed!(params[:seed])
        family_def = get_family_from_name(family_name)
        cr_def = generate_circuit_dispatch(family_def, params)
        if cr_def isa CircuitInstance
            g_def = cr_def.gates; tp_def = cr_def.t_gate_positions; nq_def = cr_def.n_qubits
        else
            g_def = cr_def.gates; tp_def = cr_def.t_positions; nq_def = cr_def.n_qubits
        end

        Random.seed!(params[:seed])
        default_result = simulate_with_gf2_screening(g_def, tp_def, nq_def;
                                                      seed=params[:seed],
                                                      amenable_indices=nothing)

        amenable_set = Set(a.t_index for a in amenability if a.is_amenable)

        Random.seed!(params[:seed])
        family_guided = get_family_from_name(family_name)
        cr_guided = generate_circuit_dispatch(family_guided, params)
        if cr_guided isa CircuitInstance
            g_guided = cr_guided.gates; tp_guided = cr_guided.t_gate_positions; nq_guided = cr_guided.n_qubits
        else
            g_guided = cr_guided.gates; tp_guided = cr_guided.t_positions; nq_guided = cr_guided.n_qubits
        end

        Random.seed!(params[:seed])
        guided_result = simulate_with_gf2_screening(g_guided, tp_guided, nq_guided;
                                                     seed=params[:seed],
                                                     amenable_indices=amenable_set)

        return (
            success = true,
            family = family_name,
            n_qubits = n_qubits_actual,
            seed = params[:seed],
            n_t_gates = n_t_total,
            gf2_rank = gf2_rank,
            nullity = nullity,
            nullity_ratio = nullity_ratio,
            n_amenable = n_amenable,
            n_not_amenable = n_not_amenable,
            amenability_rate = amenability_rate,
            default_chi = default_result.final_chi,
            default_ofd_rate = default_result.ofd_rate,
            default_time = default_result.sim_time,
            guided_chi = guided_result.final_chi,
            guided_ofd_rate = guided_result.ofd_rate,
            guided_time = guided_result.sim_time,
            n_ofd_skipped = guided_result.n_ofd_skipped,
            chi_match = default_result.final_chi == guided_result.final_chi,
            chi_ratio = default_result.final_chi > 0 ? guided_result.final_chi / default_result.final_chi : NaN,
            time_savings = default_result.sim_time > 0 ? (default_result.sim_time - guided_result.sim_time) / default_result.sim_time : NaN,
            predicted_chi = gf2_result.predicted_chi
        )
    catch e
        return (
            success = false,
            family = family_name,
            n_qubits = n,
            seed = params[:seed],
            n_t_gates = 0,
            gf2_rank = 0,
            nullity = 0,
            nullity_ratio = NaN,
            n_amenable = 0,
            n_not_amenable = 0,
            amenability_rate = NaN,
            default_chi = 0,
            default_ofd_rate = NaN,
            default_time = 0.0,
            guided_chi = 0,
            guided_ofd_rate = NaN,
            guided_time = 0.0,
            n_ofd_skipped = 0,
            chi_match = false,
            chi_ratio = NaN,
            time_savings = NaN,
            predicted_chi = 0,
            error_msg = sprint(showerror, e)
        )
    end
end

function write_results_csv(results, output_path)
    open(output_path, "w") do io
        println(io, "success,family,n_qubits,seed,n_t_gates,gf2_rank,nullity,nullity_ratio,n_amenable,n_not_amenable,amenability_rate,default_chi,default_ofd_rate,default_time,guided_chi,guided_ofd_rate,guided_time,n_ofd_skipped,chi_match,chi_ratio,time_savings,predicted_chi")

        for r in results
            println(io, join([
                r.success,
                "\"$(r.family)\"",
                r.n_qubits,
                r.seed,
                r.n_t_gates,
                r.gf2_rank,
                r.nullity,
                isnan(r.nullity_ratio) ? "" : @sprintf("%.6f", r.nullity_ratio),
                r.n_amenable,
                r.n_not_amenable,
                isnan(r.amenability_rate) ? "" : @sprintf("%.6f", r.amenability_rate),
                r.default_chi,
                isnan(r.default_ofd_rate) ? "" : @sprintf("%.6f", r.default_ofd_rate),
                @sprintf("%.4f", r.default_time),
                r.guided_chi,
                isnan(r.guided_ofd_rate) ? "" : @sprintf("%.6f", r.guided_ofd_rate),
                @sprintf("%.4f", r.guided_time),
                r.n_ofd_skipped,
                r.chi_match,
                isnan(r.chi_ratio) ? "" : @sprintf("%.6f", r.chi_ratio),
                isnan(r.time_savings) ? "" : @sprintf("%.6f", r.time_savings),
                r.predicted_chi
            ], ","))
        end
    end
end

function print_summary(results)
    successful = filter(r -> r.success, results)
    family_names = sort(unique([r.family for r in successful]))

    println()
    println("="^140)
    println("EXPERIMENT 2: GF(2)-SCREENED OFD — SUMMARY")
    println("="^140)
    @printf("%-40s %4s %5s %8s %8s %8s %8s %7s %8s\n",
            "Family", "n", "T", "Amenab%", "Def χ", "Guid χ", "χ Match", "Skipped", "TimeSav%")
    println("-"^140)

    for family in family_names
        fr = filter(r -> r.family == family, successful)
        n_sizes = sort(unique([r.n_qubits for r in fr]))

        for n in n_sizes
            nr = filter(r -> r.n_qubits == n, fr)
            if isempty(nr)
                continue
            end

            mean_t = mean([r.n_t_gates for r in nr])
            mean_amenab = mean(filter(!isnan, [r.amenability_rate for r in nr])) * 100
            mean_def_chi = mean([r.default_chi for r in nr])
            mean_guided_chi = mean([r.guided_chi for r in nr])
            n_match = count(r -> r.chi_match, nr)
            n_total_nr = length(nr)
            mean_skipped = mean([r.n_ofd_skipped for r in nr])
            time_savings_vals = filter(!isnan, [r.time_savings for r in nr])
            mean_time_sav = isempty(time_savings_vals) ? NaN : mean(time_savings_vals) * 100

            @printf("%-40s %4d %5.0f %7.1f%% %8.1f %8.1f %4d/%4d %7.1f %7.1f%%\n",
                    family, n, mean_t,
                    isnan(mean_amenab) ? 0.0 : mean_amenab,
                    mean_def_chi, mean_guided_chi,
                    n_match, n_total_nr,
                    mean_skipped,
                    isnan(mean_time_sav) ? 0.0 : mean_time_sav)
        end
    end

    n_total_succ = length(successful)
    n_chi_match = count(r -> r.chi_match, successful)
    println("="^140)
    @printf("Overall χ match rate: %d/%d (%.1f%%)\n",
            n_chi_match, n_total_succ,
            n_total_succ > 0 ? 100.0 * n_chi_match / n_total_succ : 0.0)
    println("="^140)
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("EXPERIMENT 2: GF(2)-SCREENED OFD — PRE-FILTERING VALIDATION")
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
            @printf("[%4d/%4d] FAIL %-35s n=%2d (%.1fs) — %s\n",
                    i, n_total, exp.family_name, exp.n_qubits, dt,
                    first(err_msg, 60))
        elseif i % 10 == 0 || i == n_total || dt > 2.0
            elapsed = time() - start_time
            @printf("[%4d/%4d] %-35s n=%2d t=%3d amen=%.0f%% χ_def=%d χ_guid=%d match=%s (%.1fs)\n",
                    i, n_total, exp.family_name, result.n_qubits,
                    result.n_t_gates,
                    isnan(result.amenability_rate) ? 0.0 : result.amenability_rate * 100,
                    result.default_chi, result.guided_chi,
                    result.chi_match ? "Y" : "N", dt)
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
            println("  $(r.family) n=$(r.n_qubits): $(first(msg, 80))")
        end
    end

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)

    output_csv = joinpath(output_dir, "experiment2_hybrid_scheduling.csv")
    write_results_csv(results, output_csv)
    println("\nResults saved to: $output_csv")

    print_summary(results)

    return results
end

results = main()
