# CAMPS.jl

Julia implementation of Clifford-Augmented Matrix Product States for simulating Clifford+T circuits.

## About this repo

This package implements the CAMPS representation from [Liu & Clark (arXiv:2412.17209)](https://arxiv.org/abs/2412.17209). The basic idea is to represent quantum states as:

```
|ψ⟩ = C · |ψ_MPS⟩
```

where C is a Clifford operator and |ψ_MPS⟩ is an MPS. Clifford gates are "free" — C is updated symbolically without touching the MPS. Only non-Clifford gates (like T gates) actually modify the MPS.

The Clifford tracking uses [QuantumClifford.jl](https://github.com/QuantumSavory/QuantumClifford.jl) and the MPS backend uses [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl).


## Basic usage

```julia
using CAMPS

state = CAMPSState(5)
initialize!(state)

apply_gate!(state, HGate(1))
apply_gate!(state, CNOTGate(1, 2))

apply_gate!(state, TGate(1))

amp = amplitude(state, [0, 0, 0, 0, 0])
samples = sample_state(state; num_samples=100)
```

## How it works

When a T gate (or any Rz(θ) rotation) is applied, the corresponding Pauli rotation on the MPS must be determined. Given accumulated Clifford C, applying Rz(θ) on qubit k is equivalent to applying exp(-iθ/2 · P̃) to the MPS, where P̃ = C† Z_k C is the "twisted Pauli".

The key optimization is disentangling. If the twisted Pauli has an X or Y on some qubit that is still in |0⟩, that qubit can serve as an ancilla to absorb the rotation without increasing bond dimension. This is OFD (optimization-free disentangling). When OFD fails, OBD (optimization-based disentangling) performs a variational search over two-qubit Cliffords.

### Bond dimension prediction

Given t twisted Paulis, construct a matrix M over GF(2) where M[k,j] = 1 if the k-th Pauli has X or Y on qubit j. The bond dimension scales as 2^(t - rank(M)). If the Paulis are "independent" (full rank), bond dimension stays at 1.

## Disentangling strategies

```julia
simulate_circuit(circuit, n; strategy=OFDStrategy())

simulate_circuit(circuit, n; strategy=HybridStrategy())

simulate_circuit(circuit, n; strategy=NoDisentangling())
```

## Available gates

Non-Clifford (modify MPS):
- `TGate(q)`, `TdagGate(q)`
- `RzGate(q, θ)`, `RxGate(q, θ)`, `RyGate(q, θ)`

Clifford (update tableau only):
- `HGate(q)`, `SGate(q)`, `SdagGate(q)`
- `XGate(q)`, `YGate(q)`, `ZGate(q)`
- `CNOTGate(c, t)`, `CZGate(q1, q2)`, `SWAPGate(q1, q2)`

## Running tests

```julia
using Pkg
Pkg.test("CAMPS")
```
## LLM USAGE
LLM was used for the docustrings in this repo
## References

- Liu & Clark, "Classical simulability of Clifford+T circuits with Clifford-augmented matrix product states" [arXiv:2412.17209](https://arxiv.org/abs/2412.17209)
- Qian, Huang & Qin, "Augmenting density matrix renormalization group with Clifford circuits" [Phys. Rev. Lett. 133, 190402 (2024)](https://doi.org/10.1103/PhysRevLett.133.190402)
