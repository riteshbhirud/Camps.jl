"""
Diagnose QFT N=12 bond dimension spike.
Tracks χ after every non-Clifford gate to find where OBD fails.
Also runs N=10, 14, 16 for comparison.
"""

camps_dir = dirname(dirname(@__FILE__))
using Pkg
Pkg.activate(camps_dir)

using CAMPS
using QuantumClifford
using Random
using Printf

function diagnose_qft(n_qubits::Int; verbose::Bool=true)
    circuit = qft_circuit(n_qubits)

    state = CAMPSState(n_qubits; max_bond=2048)
    initialize!(state)

    non_cliff_idx = 0
    ofd_count = 0
    obd_count = 0

    chi_trace = Int[]
    ofd_trace = Bool[]
    angle_trace = Float64[]
    qubit_trace = Int[]

    for gate in circuit
        if gate isa CliffordGate
            apply_gate!(state, gate)
        elseif gate isa RotationGate
            if is_clifford_angle(gate.angle)
                apply_gate!(state, gate)
            else
                non_cliff_idx += 1

                P_twisted = compute_twisted_pauli(state, gate.axis, gate.qubit)
                control = find_disentangling_qubit(P_twisted, state.free_qubits)
                used_ofd = control !== nothing

                if used_ofd
                    ofd_count += 1
                else
                    obd_count += 1
                end

                apply_gate!(state, gate; strategy=HybridStrategy())

                chi = get_bond_dimension(state)
                push!(chi_trace, chi)
                push!(ofd_trace, used_ofd)
                push!(angle_trace, gate.angle)
                push!(qubit_trace, gate.qubit)

                if verbose && (non_cliff_idx <= 20 || chi > 2 || non_cliff_idx % 50 == 0)
                    method = used_ofd ? "OFD" : "OBD"
                    @printf("  gate %4d: qubit=%2d, angle=%.4f, %s → χ=%d\n",
                            non_cliff_idx, gate.qubit, gate.angle, method, chi)
                end
            end
        else
            apply_gate!(state, gate)
        end
    end

    first_spike = findfirst(x -> x > 2, chi_trace)
    max_chi = maximum(chi_trace)
    max_chi_idx = argmax(chi_trace)

    return (
        n_qubits = n_qubits,
        n_non_clifford = non_cliff_idx,
        ofd_count = ofd_count,
        obd_count = obd_count,
        final_chi = chi_trace[end],
        max_chi = max_chi,
        max_chi_gate = max_chi_idx,
        first_spike_gate = first_spike,
        chi_trace = chi_trace,
        ofd_trace = ofd_trace,
        angle_trace = angle_trace,
        qubit_trace = qubit_trace
    )
end

function main()
    println("="^80)
    println("QFT N=12 BOND DIMENSION SPIKE DIAGNOSIS")
    println("="^80)

    for n in [10, 12, 14, 16]
        println("\n" * "="^80)
        println("N = $n")
        println("="^80)

        t_start = time()
        result = diagnose_qft(n; verbose=(n==12))
        elapsed = time() - t_start

        println()
        @printf("  Total non-Clifford: %d (OFD: %d, OBD: %d)\n",
                result.n_non_clifford, result.ofd_count, result.obd_count)
        @printf("  Final χ: %d\n", result.final_chi)
        @printf("  Max χ: %d (at gate %d of %d)\n",
                result.max_chi, result.max_chi_gate, result.n_non_clifford)

        if result.first_spike_gate !== nothing
            @printf("  First χ > 2: gate %d\n", result.first_spike_gate)
            spike = result.first_spike_gate
            lo = max(1, spike - 3)
            hi = min(length(result.chi_trace), spike + 5)
            println("  χ trace around first spike:")
            for i in lo:hi
                method = result.ofd_trace[i] ? "OFD" : "OBD"
                marker = i == spike ? " ← SPIKE" : ""
                @printf("    gate %4d: qubit=%2d angle=%.4f %s → χ=%d%s\n",
                        i, result.qubit_trace[i], result.angle_trace[i],
                        method, result.chi_trace[i], marker)
            end
        else
            println("  χ never exceeded 2.")
        end

        chi_counts = Dict{Int,Int}()
        for c in result.chi_trace
            chi_counts[c] = get(chi_counts, c, 0) + 1
        end
        println("  χ distribution:")
        for c in sort(collect(keys(chi_counts)))
            @printf("    χ=%4d: %d gates\n", c, chi_counts[c])
        end

        @printf("  Runtime: %.1fs\n", elapsed)
    end
end

main()
