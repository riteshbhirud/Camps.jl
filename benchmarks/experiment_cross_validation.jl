"""
Experiment: GF(2) Cross-Validation Against Benchmark Data
==========================================================

Confirms that GF(2)-predicted OFD counts match actual OFD success/failure
counts from the original 936-circuit benchmark (results_with_features.csv).

For each benchmark circuit:
  1. Regenerate the circuit from (family, n_qubits, seed, params)
  2. Run compute_gf2_for_mixed_circuit with simulate_ofd=true
  3. Compare n_disentanglable (GF(2) prediction) vs ofd_success (actual)

Expected: Perfect or near-perfect correlation for non-random families.
Random families may diverge due to :random2q RNG consumption differences
between the GF(2) Clifford walk and full MPS simulation.

Usage:
    julia benchmarks/experiment_cross_validation.jl [mode]

Modes:
    "test"   - First 50 rows only
    "medium" - All rows

Output:
    results/experiment_cross_validation.csv
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

"""
Parse a CSV line handling quoted fields (including commas inside quotes).
Returns a vector of string fields.
"""
function parse_csv_line(line::String)
    fields = String[]
    current = IOBuffer()
    in_quotes = false

    for ch in line
        if ch == '"'
            in_quotes = !in_quotes
        elseif ch == ',' && !in_quotes
            push!(fields, String(take!(current)))
        else
            write(current, ch)
        end
    end
    push!(fields, String(take!(current)))

    return fields
end

"""
Load benchmark data from results_with_features.csv.
Returns vector of NamedTuples.
"""
function load_benchmark_data(csv_path::String; max_rows::Int=0)
    lines = readlines(csv_path)
    header = parse_csv_line(lines[1])

    col_idx = Dict(h => i for (i, h) in enumerate(header))

    records = []
    for (line_num, line) in enumerate(lines[2:end])
        if max_rows > 0 && line_num > max_rows
            break
        end

        fields = parse_csv_line(line)

        success = fields[col_idx["success"]] == "true"
        if !success
            continue
        end

        family = fields[col_idx["family"]]
        n_qubits = parse(Int, fields[col_idx["n_qubits"]])
        n_t_gates = parse(Int, fields[col_idx["n_t_gates"]])
        ofd_success = parse(Int, fields[col_idx["ofd_success"]])
        ofd_fail = parse(Int, fields[col_idx["ofd_fail"]])
        seed = parse(UInt64, fields[col_idx["seed"]])

        density_str = get(fields, col_idx["density"], "")
        n_layers_str = haskey(col_idx, "n_layers") ? get(fields, col_idx["n_layers"], "") : ""

        density = isempty(density_str) ? nothing : Symbol(density_str)
        n_layers = isempty(n_layers_str) ? nothing : tryparse(Int, n_layers_str)

        push!(records, (
            family = family,
            n_qubits = n_qubits,
            n_t_gates = n_t_gates,
            ofd_success = ofd_success,
            ofd_fail = ofd_fail,
            seed = Int(seed % UInt32),
            density = density,
            n_layers = n_layers
        ))
    end

    return records
end

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

function run_cross_validation(record)
    try
        family_name = record.family
        n = record.n_qubits
        seed = record.seed

        params = Dict{Symbol, Any}(:n_qubits => n, :seed => seed)

        family = get_family_from_name(family_name)

        if family isa QFTFamily
            d = record.density !== nothing ? record.density : :medium
            params[:density] = d
        elseif family isa GroverFamily
            d = record.density !== nothing ? record.density : :half
            params[:density] = d
        elseif family isa VQEFamily
            l = record.n_layers !== nothing ? record.n_layers : 2
            params[:layers] = l
        elseif family isa SurfaceCodeFamily
            params[:n_t_gates] = record.n_t_gates
        elseif family isa QAOAMaxCutCircuit
            params[:n_t_gates] = record.n_t_gates
            params[:gamma] = π/4
            params[:beta] = π/8
        else
            params[:n_t_gates] = record.n_t_gates
            if family isa RandomBrickwallCliffordT
                params[:clifford_depth] = 2
            elseif family isa RandomAllToAllCliffordT
                params[:clifford_layers] = 2 * n
            elseif family isa DeutschJozsaCircuit
                rng_temp = Random.MersenneTwister(seed)
                params[:function_type] = rand(rng_temp, [:constant, :balanced])
            elseif family isa GraphStateCircuit
                params[:edge_probability] = 0.3
            end
        end

        Random.seed!(seed)
        circuit_result = generate_circuit_dispatch(family, params)

        if circuit_result isa CircuitInstance
            gates = circuit_result.gates
            t_positions = circuit_result.t_gate_positions
            n_qubits_actual = circuit_result.n_qubits
        else
            gates = circuit_result.gates
            t_positions = circuit_result.t_positions
            n_qubits_actual = circuit_result.n_qubits
        end

        Random.seed!(seed)
        gf2 = compute_gf2_for_mixed_circuit(
            gates, t_positions, n_qubits_actual;
            seed=seed, simulate_ofd=true)

        gf2_ofd_success = gf2.n_disentanglable
        gf2_ofd_fail = gf2.n_not_disentanglable
        actual_ofd_success = record.ofd_success
        actual_ofd_fail = record.ofd_fail

        exact_match = gf2_ofd_success == actual_ofd_success
        abs_error = abs(gf2_ofd_success - actual_ofd_success)

        return (
            success = true,
            family = family_name,
            n_qubits = n,
            seed = seed,
            n_t_gates = gf2.n_t_gates,
            actual_ofd_success = actual_ofd_success,
            actual_ofd_fail = actual_ofd_fail,
            gf2_ofd_success = gf2_ofd_success,
            gf2_ofd_fail = gf2_ofd_fail,
            gf2_rank = gf2.gf2_rank,
            gf2_nullity = gf2.nullity,
            exact_match = exact_match,
            abs_error = abs_error
        )
    catch e
        return (
            success = false,
            family = record.family,
            n_qubits = record.n_qubits,
            seed = record.seed,
            n_t_gates = 0,
            actual_ofd_success = record.ofd_success,
            actual_ofd_fail = record.ofd_fail,
            gf2_ofd_success = 0,
            gf2_ofd_fail = 0,
            gf2_rank = 0,
            gf2_nullity = 0,
            exact_match = false,
            abs_error = 0,
            error_msg = sprint(showerror, e)
        )
    end
end

function write_results_csv(results, output_path)
    open(output_path, "w") do io
        println(io, "success,family,n_qubits,seed,n_t_gates,actual_ofd_success,actual_ofd_fail,gf2_ofd_success,gf2_ofd_fail,gf2_rank,gf2_nullity,exact_match,abs_error")

        for r in results
            println(io, join([
                r.success,
                "\"$(r.family)\"",
                r.n_qubits,
                r.seed,
                r.n_t_gates,
                r.actual_ofd_success,
                r.actual_ofd_fail,
                r.gf2_ofd_success,
                r.gf2_ofd_fail,
                r.gf2_rank,
                r.gf2_nullity,
                r.exact_match,
                r.abs_error
            ], ","))
        end
    end
end

function print_summary(results)
    successful = filter(r -> r.success, results)
    family_names = sort(unique([r.family for r in successful]))

    println()
    println("="^120)
    println("CROSS-VALIDATION SUMMARY: GF(2) vs Benchmark OFD Counts")
    println("="^120)
    @printf("%-40s %6s %10s %10s %10s %8s\n",
            "Family", "Count", "Match%", "MeanAbsErr", "Correlation", "MAE/T")
    println("-"^120)

    all_actual = Float64[]
    all_predicted = Float64[]

    for family in family_names
        fr = filter(r -> r.family == family, successful)
        n_total = length(fr)
        n_match = count(r -> r.exact_match, fr)
        match_rate = n_total > 0 ? 100.0 * n_match / n_total : 0.0
        mean_abs_err = mean([r.abs_error for r in fr])
        mean_t = mean([r.n_t_gates for r in fr])
        mae_per_t = mean_t > 0 ? mean_abs_err / mean_t : NaN

        actual = Float64.([r.actual_ofd_success for r in fr])
        predicted = Float64.([r.gf2_ofd_success for r in fr])
        append!(all_actual, actual)
        append!(all_predicted, predicted)

        if length(actual) >= 2 && std(actual) > 0 && std(predicted) > 0
            corr = cor(actual, predicted)
        else
            corr = NaN
        end

        @printf("%-40s %6d %9.1f%% %10.2f %10.4f %8.4f\n",
                family, n_total, match_rate, mean_abs_err,
                isnan(corr) ? 0.0 : corr, isnan(mae_per_t) ? 0.0 : mae_per_t)
    end

    n_total = length(successful)
    n_match = count(r -> r.exact_match, successful)
    overall_match = n_total > 0 ? 100.0 * n_match / n_total : 0.0
    overall_mae = mean([r.abs_error for r in successful])

    if length(all_actual) >= 2 && std(all_actual) > 0 && std(all_predicted) > 0
        overall_corr = cor(all_actual, all_predicted)
        overall_r2 = overall_corr^2
    else
        overall_corr = NaN
        overall_r2 = NaN
    end

    println("="^120)
    @printf("OVERALL: %d circuits, exact match = %d (%.1f%%), MAE = %.2f, R = %.4f, R² = %.4f\n",
            n_total, n_match, overall_match, overall_mae,
            isnan(overall_corr) ? 0.0 : overall_corr,
            isnan(overall_r2) ? 0.0 : overall_r2)
    println("="^120)
end

function main()
    mode = length(ARGS) >= 1 ? ARGS[1] : "medium"

    println("="^80)
    println("GF(2) CROSS-VALIDATION AGAINST BENCHMARK DATA")
    println("="^80)
    println("Mode: $mode")
    println("Date: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
    println()

    csv_path = joinpath(camps_dir, "results", "results_with_features.csv")
    max_rows = mode == "test" ? 50 : 0
    records = load_benchmark_data(csv_path; max_rows=max_rows)
    println("Loaded $(length(records)) benchmark circuits")
    println()

    println("Running GF(2) analysis on each circuit...")
    println("-"^80)

    start_time = time()
    results = []

    for (i, record) in enumerate(records)
        t0 = time()
        result = run_cross_validation(record)
        dt = time() - t0
        push!(results, result)

        if !result.success
            err_msg = hasproperty(result, :error_msg) ? result.error_msg : "unknown"
            @printf("[%4d/%4d] FAIL %-35s n=%2d (%.1fs) — %s\n",
                    i, length(records), record.family, record.n_qubits, dt,
                    first(err_msg, 60))
        elseif i % 50 == 0 || i == length(records) || dt > 2.0
            @printf("[%4d/%4d] %-35s n=%2d t=%3d actual=%3d gf2=%3d match=%s (%.1fs)\n",
                    i, length(records), result.family, result.n_qubits,
                    result.n_t_gates, result.actual_ofd_success,
                    result.gf2_ofd_success, result.exact_match ? "Y" : "N", dt)
        end
    end

    total_time = time() - start_time
    println("-"^80)
    @printf("Completed %d cross-validations in %.1f seconds\n", length(records), total_time)

    n_success = count(r -> r.success, results)
    n_failed = length(results) - n_success
    println("Success: $n_success, Failed: $n_failed")

    output_dir = joinpath(camps_dir, "results")
    mkpath(output_dir)

    output_csv = joinpath(output_dir, "experiment_cross_validation.csv")
    write_results_csv(results, output_csv)
    println("\nResults saved to: $output_csv")

    print_summary(results)

    return results
end

results = main()
