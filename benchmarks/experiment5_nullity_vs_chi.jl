"""
Experiment 5: Nullity vs Bond Dimension Validation
===================================================

Validates the core theoretical predictions from Theorem 1:

  Claim 1: There EXISTS an ordering achieving m ≥ rank(z) OFD successes.
  Claim 2: For ANY ordering with m OFD successes, χ ≤ 2^(t − m).

Two bounds are validated:

  (A) Structural bound: χ_OFD ≤ 2^(t − m_OFD)  [Claim 2, must always hold]
  (B) Optimal bound:    χ_OFD ≤ 2^ν = 2^(t − rank)  [holds when greedy achieves m ≥ rank]

For each circuit, we:
1. Compute GF(2) nullity ν via Clifford-only walk (cheap)
2. Run full CAMPS simulation with greedy OFD to get χ_OFD and m_OFD
3. Run full CAMPS simulation with NoDisentangling for baseline χ_naive
4. Validate both bounds and report optimality (m_OFD vs rank)

Greedy OFD processes T-gates in circuit order, picking the first available
free qubit with x-bit=1. This may "waste" a qubit needed by a later gate,
so greedy OFD can achieve m < rank(z) in some circuits. The structural
bound χ ≤ 2^(t − m) still holds — it just gives a weaker guarantee than
the optimal 2^ν when m < rank.

Usage:
    julia benchmarks/experiment5_nullity_vs_chi.jl [mode]

Modes:
    "test"   - Quick validation (n=4, few families)
    "medium" - Standard (n=4..8, all families, multiple seeds)

Output:
    results/experiment5_nullity_vs_chi.csv
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
    simulate_and_measure(gates, t_positions, n_qubits, strategy; seed=nothing)

Run full CAMPS simulation and return bond dimension + entropy metrics.
Works with both tuple-format and Gate-object circuits.
"""
function simulate_and_measure(gates::Vector, t_positions::Vector{Int},
                               n_qubits::Int, strategy::DisentanglingStrategy;
                               seed::Union{Integer,Nothing}=nothing,
                               max_bond::Int=512, cutoff::Float64=1e-14)
    if seed !== nothing
        Random.seed!(seed)
    end

    state = CAMPSState(n_qubits; max_bond=max_bond, cutoff=cutoff)
    initialize!(state)

    n_ofd = 0
    n_absorbed = 0

    for gate in gates
        if gate isa Tuple
            gate_type = gate[1]
            qubits = gate[2]

            if gate_type == :T
                qubit = qubits[1]

                P_twisted = compute_twisted_pauli(state, :Z, qubit)
                control = find_disentangling_qubit(P_twisted, state.free_qubits)
                if control !== nothing && strategy isa OFDStrategy
                    n_ofd += 1
                else
                    n_absorbed += 1
                end

                apply_gate!(state, TGate(qubit); strategy=strategy)

            elseif gate_type == :H
                apply_gate!(state, CliffordGate([(:H, qubits[1])], [qubits[1]]); strategy=strategy)
            elseif gate_type == :CNOT
                apply_gate!(state, CliffordGate([(:CNOT, qubits[1], qubits[2])], [qubits[1], qubits[2]]); strategy=strategy)
            elseif gate_type == :X
                apply_gate!(state, CliffordGate([(:X, qubits[1])], [qubits[1]]); strategy=strategy)
            elseif gate_type == :Z
                apply_gate!(state, CliffordGate([(:Z, qubits[1])], [qubits[1]]); strategy=strategy)
            elseif gate_type == :S
                apply_gate!(state, CliffordGate([(:S, qubits[1])], [qubits[1]]); strategy=strategy)
            elseif gate_type == :Sdag
                apply_gate!(state, CliffordGate([(:Sdag, qubits[1])], [qubits[1]]); strategy=strategy)
            elseif gate_type == :random2q
                q1, q2 = qubits[1], qubits[2]
                cliff = random_clifford(2)
                sparse = SparseGate(cliff, [q1, q2])
                apply!(state.clifford, sparse)
            end

        elseif gate isa CliffordGate
            apply_gate!(state, gate; strategy=strategy)

        elseif gate isa RotationGate
            if !is_clifford_angle(gate.angle) && strategy isa OFDStrategy
                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                control = find_disentangling_qubit(P_twisted, state.free_qubits)
                if control !== nothing
                    n_ofd += 1
                else
                    n_absorbed += 1
                end
            elseif !is_clifford_angle(gate.angle)
                n_absorbed += 1
            end
            apply_gate!(state, gate; strategy=strategy)
        end
    end

    final_chi = get_bond_dimension(state)
    max_entropy = max_entanglement_entropy(state.mps)

    return (
        final_chi = final_chi,
        max_entropy = max_entropy,
        n_ofd = n_ofd,
        n_absorbed = n_absorbed,
        ofd_rate = (n_ofd + n_absorbed) > 0 ? n_ofd / (n_ofd + n_absorbed) : NaN
    )
end

function generate_experiments_exp5(mode::String)
    experiments = []

    n_range = mode == "test" ? [4, 5] : [4, 5, 6, 7, 8]
    n_realizations = mode == "test" ? 2 : 5

    families = [
        ("Random Clifford+T (Brick-wall)", :brick),
        ("Random Clifford+T (All-to-all)", :a2a),
        ("Bernstein-Vazirani", :bv),
        ("Simon's Algorithm", :simon),
        ("Deutsch-Jozsa", :dj),
        ("GHZ State", :ghz),
        ("Bell State / EPR Pairs", :bell),
        ("Graph State", :graph),
        ("Cluster State (1D)", :cluster),
        ("QAOA MaxCut (p=1, 3-regular)", :qaoa),
        ("Surface Code", :surface),
        ("Quantum Fourier Transform", :qft),
        ("Grover Search", :grover),
        ("VQE Hardware-Efficient Ansatz", :vqe),
    ]

    for (family_name, tag) in families
        for n in n_range
            if tag == :simon && n % 2 != 0
                continue
            end
            if tag == :qaoa && n % 2 != 0
                continue
            end

            for real in 1:n_realizations
                seed = Int(hash(("exp5", tag, n, real)) % UInt32)

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
                elseif tag == :surface
                    params[:n_t_gates] = max(4, div(n, 2))
                elseif tag == :qft
                    params[:density] = :medium
                elseif tag == :grover
                    params[:density] = :half
                elseif tag == :vqe
                    params[:layers] = 2
                end

                push!(experiments, (
                    family_name = family_name,
                    n_qubits = n,
                    tag = tag,
                    params = params
                ))
            end
        end
    end

    return experiments
end

function run_exp5(experiment)
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
        gf2 = compute_gf2_for_mixed_circuit(
            gates, t_positions, n_qubits_actual;
            seed=params[:seed], simulate_ofd=true)

        gf2_rank_val = gf2.gf2_rank
        nullity = gf2.nullity
        n_t = gf2.n_t_gates
        predicted_chi = gf2.predicted_chi

        Random.seed!(params[:seed])
        family_sim1 = get_family_from_name(family_name)
        cr_sim1 = generate_circuit_dispatch(family_sim1, params)
        if cr_sim1 isa CircuitInstance
            g1 = cr_sim1.gates; tp1 = cr_sim1.t_gate_positions; nq1 = cr_sim1.n_qubits
        else
            g1 = cr_sim1.gates; tp1 = cr_sim1.t_positions; nq1 = cr_sim1.n_qubits
        end

        Random.seed!(params[:seed])
        ofd_sim = simulate_and_measure(g1, tp1, nq1, OFDStrategy(); seed=params[:seed])

        Random.seed!(params[:seed])
        family_sim2 = get_family_from_name(family_name)
        cr_sim2 = generate_circuit_dispatch(family_sim2, params)
        if cr_sim2 isa CircuitInstance
            g2 = cr_sim2.gates; tp2 = cr_sim2.t_gate_positions; nq2 = cr_sim2.n_qubits
        else
            g2 = cr_sim2.gates; tp2 = cr_sim2.t_positions; nq2 = cr_sim2.n_qubits
        end

        Random.seed!(params[:seed])
        no_dis_sim = simulate_and_measure(g2, tp2, nq2, NoDisentangling(); seed=params[:seed])

        m_ofd = ofd_sim.n_ofd

        structural_exponent = n_t - m_ofd
        structural_bound = structural_exponent < 64 ? 2^structural_exponent : typemax(Int)
        structural_satisfied = ofd_sim.final_chi <= structural_bound

        optimal_satisfied = ofd_sim.final_chi <= predicted_chi

        greedy_optimal = m_ofd >= gf2_rank_val

        tightness = structural_exponent > 0 ? log2(max(1, ofd_sim.final_chi)) / structural_exponent : (ofd_sim.final_chi <= 1 ? 1.0 : NaN)

        return (
            success = true,
            family = family_name,
            n_qubits = n_qubits_actual,
            seed = params[:seed],
            n_t_gates = n_t,
            gf2_rank = gf2_rank_val,
            nullity = nullity,
            predicted_chi = predicted_chi,
            ofd_chi = ofd_sim.final_chi,
            ofd_entropy = ofd_sim.max_entropy,
            m_ofd = m_ofd,
            ofd_rate = ofd_sim.ofd_rate,
            nodis_chi = no_dis_sim.final_chi,
            nodis_entropy = no_dis_sim.max_entropy,
            structural_bound = structural_bound,
            structural_satisfied = structural_satisfied,
            optimal_satisfied = optimal_satisfied,
            greedy_optimal = greedy_optimal,
            tightness = tightness,
            chi_reduction = no_dis_sim.final_chi > 0 ? ofd_sim.final_chi / no_dis_sim.final_chi : NaN,
            log2_chi_ofd = log2(max(1, ofd_sim.final_chi)),
            log2_chi_nodis = log2(max(1, no_dis_sim.final_chi)),
            log2_predicted = nullity > 0 ? Float64(nullity) : 0.0,
            log2_structural = Float64(structural_exponent)
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
            predicted_chi = 0,
            ofd_chi = 0,
            ofd_entropy = 0.0,
            m_ofd = 0,
            ofd_rate = NaN,
            nodis_chi = 0,
            nodis_entropy = 0.0,
            structural_bound = 0,
            structural_satisfied = false,
            optimal_satisfied = false,
            greedy_optimal = false,
            tightness = NaN,
            chi_reduction = NaN,
            log2_chi_ofd = 0.0,
            log2_chi_nodis = 0.0,
            log2_predicted = 0.0,
            log2_structural = 0.0,
            error_msg = sprint(showerror, e)
        )
    end
end

function write_results_csv_exp5(results, output_path)
    open(output_path, "w") do io
        println(io, "success,family,n_qubits,seed,n_t_gates,gf2_rank,nullity,predicted_chi,ofd_chi,ofd_entropy,m_ofd,ofd_rate,nodis_chi,nodis_entropy,structural_bound,structural_satisfied,optimal_satisfied,greedy_optimal,tightness,chi_reduction,log2_chi_ofd,log2_chi_nodis,log2_predicted,log2_structural")

        for r in results
            println(io, join([
                r.success,
                "\"$(r.family)\"",
                r.n_qubits,
                r.seed,
                r.n_t_gates,
                r.gf2_rank,
                r.nullity,
                r.predicted_chi,
                r.ofd_chi,
                @sprintf("%.6f", r.ofd_entropy),
                r.m_ofd,
                isnan(r.ofd_rate) ? "" : @sprintf("%.6f", r.ofd_rate),
                r.nodis_chi,
                @sprintf("%.6f", r.nodis_entropy),
                r.structural_bound,
                r.structural_satisfied,
                r.optimal_satisfied,
                r.greedy_optimal,
                isnan(r.tightness) ? "" : @sprintf("%.6f", r.tightness),
                isnan(r.chi_reduction) ? "" : @sprintf("%.6f", r.chi_reduction),
                @sprintf("%.4f", r.log2_chi_ofd),
                @sprintf("%.4f", r.log2_chi_nodis),
                @sprintf("%.4f", r.log2_predicted),
                @sprintf("%.4f", r.log2_structural)
            ], ","))
        end
    end
end

function print_summary_exp5(results)
    successful = filter(r -> r.success, results)
    family_names = sort(unique([r.family for r in successful]))

    println()
    println("="^160)
    println("EXPERIMENT 5: NULLITY vs BOND DIMENSION VALIDATION — SUMMARY")
    println("="^160)
    @printf("%-40s %4s %4s %4s %5s %5s %8s %8s %8s %8s %6s %6s %6s\n",
            "Family", "n", "T", "rank", "m", "ν", "χ_pred", "χ_OFD", "χ_NoDis", "χ_struc", "Struc", "Opt", "Grdy=")
    println("-"^160)

    n_structural_ok = 0
    n_optimal_ok = 0
    n_greedy_optimal = 0
    n_total = 0

    for family in family_names
        fr = filter(r -> r.family == family, successful)
        n_sizes = sort(unique([r.n_qubits for r in fr]))

        for n in n_sizes
            nr = filter(r -> r.n_qubits == n, fr)
            if isempty(nr)
                continue
            end

            mean_t = mean([r.n_t_gates for r in nr])
            mean_rank = mean([r.gf2_rank for r in nr])
            mean_m = mean([r.m_ofd for r in nr])
            mean_null = mean([r.nullity for r in nr])
            mean_pred = mean([r.predicted_chi for r in nr])
            mean_ofd = mean([r.ofd_chi for r in nr])
            mean_nodis = mean([r.nodis_chi for r in nr])
            mean_struc = mean([r.structural_bound for r in nr])
            struc_ok = all(r -> r.structural_satisfied, nr)
            opt_ok = all(r -> r.optimal_satisfied, nr)
            grdy_opt = all(r -> r.greedy_optimal, nr)

            n_total += length(nr)
            n_structural_ok += count(r -> r.structural_satisfied, nr)
            n_optimal_ok += count(r -> r.optimal_satisfied, nr)
            n_greedy_optimal += count(r -> r.greedy_optimal, nr)

            @printf("%-40s %4d %4.0f %4.0f %5.0f %5.1f %8.0f %8.1f %8.1f %8.0f %6s %6s %6s\n",
                    family, n, mean_t, mean_rank, mean_m, mean_null,
                    mean_pred, mean_ofd, mean_nodis, mean_struc,
                    struc_ok ? "✓" : "✗",
                    opt_ok ? "✓" : "✗",
                    grdy_opt ? "✓" : "✗")
        end
    end

    println("-"^160)
    println()
    println("BOUND VALIDATION:")
    println("  Structural bound χ ≤ 2^(t−m):  $n_structural_ok / $n_total ($(round(100*n_structural_ok/max(1,n_total), digits=1))%)  [Theorem 1, Claim 2 — must always hold]")
    println("  Optimal bound    χ ≤ 2^ν:       $n_optimal_ok / $n_total ($(round(100*n_optimal_ok/max(1,n_total), digits=1))%)  [requires greedy m ≥ rank]")
    println()
    println("GREEDY OPTIMALITY:")
    println("  Greedy achieves m ≥ rank:        $n_greedy_optimal / $n_total ($(round(100*n_greedy_optimal/max(1,n_total), digits=1))%)")
    n_suboptimal = n_total - n_greedy_optimal
    if n_suboptimal > 0
        println("  Greedy suboptimal (m < rank):    $n_suboptimal / $n_total — these circuits need non-greedy ordering for optimal bound")
        subopt_families = unique([r.family for r in successful if !r.greedy_optimal])
        for fam in subopt_families
            sr = filter(r -> r.family == fam && !r.greedy_optimal, successful)
            deficits = [r.gf2_rank - r.m_ofd for r in sr]
            println("    $fam: $(length(sr)) cases, rank−m deficit = $(minimum(deficits))..$(maximum(deficits))")
        end
    end

    println()
    println("OFD EFFECTIVENESS:")
    families_with_t = filter(r -> r.n_t_gates > 0, successful)
    if !isempty(families_with_t)
        mean_reduction = mean(filter(!isnan, [r.chi_reduction for r in families_with_t]))
        println("  Mean χ_OFD / χ_NoDis: $(@sprintf("%.4f", mean_reduction)) (lower = more effective)")
    end
    println("="^160)
end

function main_exp5()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("EXPERIMENT 5: NULLITY vs BOND DIMENSION VALIDATION")
    println("="^80)
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    experiments = generate_experiments_exp5(mode)
    n_total = length(experiments)
    println("Total experiments: $n_total")

    families = unique([e.family_name for e in experiments])
    for fam in sort(families)
        fe = filter(e -> e.family_name == fam, experiments)
        n_sizes = sort(unique([e.n_qubits for e in fe]))
        println("  $fam: $(length(fe)) experiments, n ∈ {$(join(n_sizes, ", "))}")
    end
    println()

    println("Starting full simulations (requires MPS)...")
    println("-"^80)

    start_time = time()
    results = []

    for (i, exp) in enumerate(experiments)
        t0 = time()
        result = run_exp5(exp)
        dt = time() - t0
        push!(results, result)

        if !result.success
            err_msg = hasproperty(result, :error_msg) ? result.error_msg : "unknown"
            @printf("[%4d/%4d] FAIL %-35s n=%2d (%.1fs) — %s\n",
                    i, n_total, exp.family_name, exp.n_qubits, dt,
                    first(err_msg, 60))
        elseif i % 5 == 0 || i == n_total || dt > 5.0
            struc_str = result.structural_satisfied ? "S✓" : "S✗"
            opt_str = result.optimal_satisfied ? "O✓" : "O✗"
            @printf("[%4d/%4d] %-35s n=%2d t=%2d rank=%2d m=%2d ν=%2d χ=%4d %s %s (%.1fs)\n",
                    i, n_total, exp.family_name, result.n_qubits,
                    result.n_t_gates, result.gf2_rank, result.m_ofd,
                    result.nullity, result.ofd_chi,
                    struc_str, opt_str, dt)
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

    output_csv = joinpath(output_dir, "experiment5_nullity_vs_chi.csv")
    write_results_csv_exp5(results, output_csv)
    println("\nResults saved to: $output_csv")

    print_summary_exp5(results)

    return results
end

results = main_exp5()
