
"""
    PauliExpectations

Precomputed Pauli expectations for a two-site reduced density matrix.

For a two-qubit state ρ, stores ⟨σ₁⊗σ₂⟩ = Tr((σ₁⊗σ₂)ρ) for all 16 combinations
of σ₁, σ₂ ∈ {I, X, Y, Z}.

The expectations are stored as a 4×4 matrix indexed by Pauli indices 1-4.
"""
struct PauliExpectations
    values::Matrix{ComplexF64}
end

"""
    PAULI_INDEX

Map from Pauli symbol to index: I=1, X=2, Y=3, Z=4
"""
const PAULI_INDEX = Dict(:I => 1, :X => 2, :Y => 3, :Z => 4)
const INDEX_TO_PAULI = [:I, :X, :Y, :Z]

"""
    CliffordPauliMap

Precomputed mapping of how a Clifford transforms (σ₁⊗I) for the 4 Paulis.

For OBD, U†(σ₁⊗I)U is needed for each σ₁ ∈ {I, X, Y, Z}.
Since U is Clifford, U†PU = ±Q for some Pauli Q.

Stores:
- mapped_paulis: 4-element vector of (σ₁_idx, σ₂_idx) tuples
- phases: 4-element vector of phase factors (+1, -1, +i, -i)
"""
struct CliffordPauliMap
    mapped_paulis::Vector{Tuple{Int, Int}}
    phases::Vector{ComplexF64}
end

const CLIFFORD_PAULI_MAP_CACHE = Ref{Union{Vector{CliffordPauliMap}, Nothing}}(nothing)
const CLIFFORD_PAULI_MAP_CACHE_LOCK = ReentrantLock()

"""
    precompute_pauli_expectations(rho::Matrix{ComplexF64}) -> PauliExpectations

Compute all 16 Pauli expectations ⟨σ₁⊗σ₂⟩ = Tr((σ₁⊗σ₂)ρ) for a 4×4 density matrix.

# Arguments
- `rho::Matrix{ComplexF64}`: 4×4 two-qubit density matrix

# Returns
- `PauliExpectations`: Structure containing all 16 expectation values

# Complexity
O(16 × 16) = O(256) since each expectation is a trace of 4×4 matrices.

# Note
Uses the property that Tr(Pρ) = Σᵢⱼ Pᵢⱼ ρⱼᵢ for efficient computation.
"""
function precompute_pauli_expectations(rho::Matrix{ComplexF64})::PauliExpectations
    I2 = ComplexF64[1 0; 0 1]
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    paulis = [I2, X, Y, Z]

    expectations = zeros(ComplexF64, 4, 4)

    for (i, σ1) in enumerate(paulis)
        for (j, σ2) in enumerate(paulis)
            P = kron(σ1, σ2)
            expectations[i, j] = tr(P * rho)
        end
    end

    return PauliExpectations(expectations)
end

"""
    compute_clifford_pauli_action(U::Matrix{ComplexF64}, σ_idx::Int) -> Tuple{Int, Int, ComplexF64}

Compute how a Clifford U transforms (σ⊗I): U†(σ⊗I)U = phase × (τ₁⊗τ₂)

# Arguments
- `U::Matrix{ComplexF64}`: 4×4 Clifford unitary matrix
- `σ_idx::Int`: Index of input Pauli (1=I, 2=X, 3=Y, 4=Z)

# Returns
- `Tuple{Int, Int, ComplexF64}`: (τ₁_idx, τ₂_idx, phase) where U†(σ⊗I)U = phase×(τ₁⊗τ₂)

# Note
The phase is one of {+1, -1, +i, -i} since Cliffords map Paulis to Paulis.
"""
function compute_clifford_pauli_action(U::Matrix{ComplexF64}, σ_idx::Int)::Tuple{Int, Int, ComplexF64}
    I2 = ComplexF64[1 0; 0 1]
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    paulis = [I2, X, Y, Z]

    σ = paulis[σ_idx]
    P_in = kron(σ, I2)

    P_out = U' * P_in * U

    best_match = (1, 1)
    best_phase = 1.0 + 0.0im
    best_score = 0.0

    for (i, τ1) in enumerate(paulis)
        for (j, τ2) in enumerate(paulis)
            P_candidate = kron(τ1, τ2)
            coeff = tr(P_candidate * P_out) / 4
            score = abs(coeff)
            if score > best_score
                best_score = score
                best_match = (i, j)
                if abs(coeff - 1.0) < 1e-10
                    best_phase = 1.0 + 0.0im
                elseif abs(coeff + 1.0) < 1e-10
                    best_phase = -1.0 + 0.0im
                elseif abs(coeff - im) < 1e-10
                    best_phase = 0.0 + 1.0im
                elseif abs(coeff + im) < 1e-10
                    best_phase = 0.0 - 1.0im
                else
                    best_phase = coeff / abs(coeff)
                end
            end
        end
    end

    return (best_match[1], best_match[2], best_phase)
end

"""
    build_clifford_pauli_map(U::Matrix{ComplexF64}) -> CliffordPauliMap

Build the Pauli mapping for a single Clifford unitary.

# Arguments
- `U::Matrix{ComplexF64}`: 4×4 Clifford unitary matrix

# Returns
- `CliffordPauliMap`: Mapping of (σ⊗I) → U†(σ⊗I)U for σ ∈ {I, X, Y, Z}
"""
function build_clifford_pauli_map(U::Matrix{ComplexF64})::CliffordPauliMap
    mapped_paulis = Vector{Tuple{Int, Int}}(undef, 4)
    phases = Vector{ComplexF64}(undef, 4)

    for σ_idx in 1:4
        τ1_idx, τ2_idx, phase = compute_clifford_pauli_action(U, σ_idx)
        mapped_paulis[σ_idx] = (τ1_idx, τ2_idx)
        phases[σ_idx] = phase
    end

    return CliffordPauliMap(mapped_paulis, phases)
end

"""
    get_or_build_clifford_pauli_cache(cache::Union{TwoQubitCliffordCache, Nothing}=nothing) -> Vector{CliffordPauliMap}

Get or build the global cache of Clifford-Pauli mappings.

This cache is built once and reused for all OBD evaluations.
Thread-safe via lock protection.

# Arguments
- `cache::Union{TwoQubitCliffordCache, Nothing}`: Optional precomputed Clifford cache

# Returns
- `Vector{CliffordPauliMap}`: Vector of 11,520 Pauli mappings (one per Clifford)
"""
function get_or_build_clifford_pauli_cache(cache::Union{TwoQubitCliffordCache, Nothing}=nothing)::Vector{CliffordPauliMap}
    lock(CLIFFORD_PAULI_MAP_CACHE_LOCK) do
        if CLIFFORD_PAULI_MAP_CACHE[] !== nothing
            return CLIFFORD_PAULI_MAP_CACHE[]::Vector{CliffordPauliMap}
        end

        if cache !== nothing
            matrices = cache.matrices
        else
            cliffords = get_all_two_qubit_cliffords()
            matrices = [clifford_to_matrix(C) for C in cliffords]
        end

        n_clif = length(matrices)
        pauli_maps = Vector{CliffordPauliMap}(undef, n_clif)

        for i in 1:n_clif
            pauli_maps[i] = build_clifford_pauli_map(matrices[i])
        end

        CLIFFORD_PAULI_MAP_CACHE[] = pauli_maps
        return pauli_maps
    end
end

"""
    evaluate_renyi2_from_pauli(expectations::PauliExpectations, pauli_map::CliffordPauliMap) -> Float64

Evaluate the second Rényi entropy after applying a Clifford, using precomputed Pauli data.

# Formula
For ρ' = UρU† and ρ₁ = Tr₂(ρ'):
    Tr(ρ₁²) = (1/2) Σ_{σ₁∈{I,X,Y,Z}} |⟨U†(σ₁⊗I)U⟩_ρ|²

Since U†(σ₁⊗I)U = phase × (τ₁⊗τ₂), this gives:
    ⟨U†(σ₁⊗I)U⟩_ρ = phase × ⟨τ₁⊗τ₂⟩_ρ

And |phase|² = 1, so:
    Tr(ρ₁²) = (1/2) Σ_{σ₁} |⟨τ₁⊗τ₂⟩_ρ|²

# Arguments
- `expectations::PauliExpectations`: Precomputed Pauli expectations
- `pauli_map::CliffordPauliMap`: Precomputed Clifford-Pauli mapping

# Returns
- `Float64`: Second Rényi entropy S₂ = -log(Tr(ρ₁²))

# Complexity
O(4) - just 4 lookups and arithmetic operations.
"""
function evaluate_renyi2_from_pauli(expectations::PauliExpectations, pauli_map::CliffordPauliMap)::Float64
    purity = 0.0

    for σ_idx in 1:4
        τ1_idx, τ2_idx = pauli_map.mapped_paulis[σ_idx]
        exp_val = expectations.values[τ1_idx, τ2_idx]
        purity += abs2(exp_val)
    end

    purity_normalized = purity / 2.0

    purity_normalized = clamp(real(purity_normalized), 1e-15, 1.0)

    return -log(purity_normalized)
end

"""
    find_optimal_clifford_fast(expectations::PauliExpectations,
                                pauli_maps::Vector{CliffordPauliMap};
                                use_representatives::Bool=false) -> Tuple{Int, Float64}

Find the optimal Clifford to minimize Rényi-2 entropy using fast O(1) evaluation.

# Arguments
- `expectations::PauliExpectations`: Precomputed Pauli expectations from RDM
- `pauli_maps::Vector{CliffordPauliMap}`: Precomputed Clifford-Pauli mappings
- `use_representatives::Bool`: If true, only search over representative subset

# Returns
- `Tuple{Int, Float64}`: (best_index, best_entropy)

# Complexity
O(11520 × 4) = O(46080) ≈ O(1) per Clifford
This is ~30× faster than the naive O(11520 × 128) approach.
"""
function find_optimal_clifford_fast(expectations::PauliExpectations,
                                     pauli_maps::Vector{CliffordPauliMap};
                                     use_representatives::Bool=false)::Tuple{Int, Float64}
    n_search = use_representatives ? min(length(pauli_maps), 20) : length(pauli_maps)

    best_idx = 1
    best_entropy = Inf

    for i in 1:n_search
        entropy = evaluate_renyi2_from_pauli(expectations, pauli_maps[i])
        if entropy < best_entropy
            best_entropy = entropy
            best_idx = i
        end
    end

    return (best_idx, best_entropy)
end

"""
    Renyi2BaseTensor

Base tensor for Equation 19 purity computation.

The tensor T[a,b,c,d,e,f,g,h] = ψ[a,e] × ψ*[b,f] × ψ[c,g] × ψ*[d,h]
where ψ is the two-site wavefunction and indices are:
- a,b: site n indices for copy 1 (ket, bra)
- c,d: site n indices for copy 2 (ket, bra)
- e,f: site n+1 indices for copy 1 (ket, bra)
- g,h: site n+1 indices for copy 2 (ket, bra)

This 256-element tensor encodes all information needed to compute
Tr(ρ_n²) for any two-qubit Clifford U applied to the state.
"""
struct Renyi2BaseTensor
    values::Array{ComplexF64, 8}
end

"""
    Renyi2ContractionKernel

Precomputed contraction kernel for a Clifford unitary.

For Clifford U, the kernel K[a,b,c,d,e,f,g,h] encodes how to contract
with the base tensor T to compute Tr(ρ_n'²) after applying U.

The purity is: Tr(ρ_n'²) = Σ T[a,b,c,d,e,f,g,h] × K[a,b,c,d,e,f,g,h]
"""
struct Renyi2ContractionKernel
    values::Array{ComplexF64, 8}
end

const RENYI2_KERNEL_CACHE = Ref{Union{Vector{Renyi2ContractionKernel}, Nothing}}(nothing)
const RENYI2_KERNEL_CACHE_LOCK = ReentrantLock()

"""
    precompute_renyi2_base_tensor(mps::MPS, bond::Int, sites) -> Renyi2BaseTensor

Compute the 256-element base tensor for Equation 19 purity evaluation.

This computes the base tensor from the two-site reduced density matrix,
which properly handles link indices in the MPS.

# Arguments
- `mps::MPS`: Matrix Product State (not modified)
- `bond::Int`: Bond index (between sites bond and bond+1)
- `sites`: Site indices (unused, kept for API compatibility)

# Returns
- `Renyi2BaseTensor`: 256-element base tensor T[a,b,c,d,e,f,g,h]

# Complexity
O(χ³) for extracting the two-site RDM, O(256) for building T.

# Formula
T[a,b,c,d,e,f,g,h] = ρ[a,e; b,f] × ρ[c,g; d,h]
where ρ is the two-site reduced density matrix.

This is equivalent to ψ[a,e] × ψ*[b,f] × ψ[c,g] × ψ*[d,h] when ρ = |ψ⟩⟨ψ|.
"""
function precompute_renyi2_base_tensor(mps::MPS, bond::Int, sites)::Renyi2BaseTensor
    site1 = bond
    site2 = bond + 1

    rho = extract_two_site_rdm(mps, site1, site2)

    T = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2)

    for a in 1:2, b in 1:2, c in 1:2, d in 1:2
        for e in 1:2, f in 1:2, g in 1:2, h in 1:2
            row1 = 2*(a-1) + e
            col1 = 2*(b-1) + f
            row2 = 2*(c-1) + g
            col2 = 2*(d-1) + h
            T[a,b,c,d,e,f,g,h] = rho[row1, col1] * rho[row2, col2]
        end
    end

    return Renyi2BaseTensor(T)
end

"""
    compute_renyi2_kernel(U::Matrix{ComplexF64}) -> Renyi2ContractionKernel

Compute the contraction kernel for a Clifford unitary.

The kernel encodes the index contractions in the purity diagram:
- Partial trace over site n+1 (e'=f' and g'=h')
- Matrix multiplication and trace for Tr(ρ_n²) (b'=c' and a'=d')

# Arguments
- `U::Matrix{ComplexF64}`: 4×4 Clifford unitary matrix

# Returns
- `Renyi2ContractionKernel`: 256-element contraction kernel

# Formula
K[a,b,c,d,e,f,g,h] = Σ_{k,m,p,q} U[k,p,a,e] × U*[m,p,b,f] × U[m,q,c,g] × U*[k,q,d,h]
where the sum runs over all values of (k,m,p,q) ∈ {1,2}^4 that satisfy:
- k = a' = d' (trace condition)
- m = b' = c' (matrix multiplication)
- p = e' = f' (partial trace copy 1)
- q = g' = h' (partial trace copy 2)

# Complexity
O(256) - 16 terms in the sum for each of 256 output elements.
"""
function compute_renyi2_kernel(U::Matrix{ComplexF64})::Renyi2ContractionKernel
    @assert size(U) == (4, 4) "U must be a 4×4 matrix"

    U_tensor = zeros(ComplexF64, 2, 2, 2, 2)
    for k in 1:2, p in 1:2, a in 1:2, e in 1:2
        row = 2*(k-1) + p
        col = 2*(a-1) + e
        U_tensor[k, p, a, e] = U[row, col]
    end

    K = zeros(ComplexF64, 2, 2, 2, 2, 2, 2, 2, 2)

    for a in 1:2, b in 1:2, c in 1:2, d in 1:2
        for e in 1:2, f in 1:2, g in 1:2, h in 1:2
            val = zero(ComplexF64)
            for k in 1:2, m in 1:2, p in 1:2, q in 1:2
                val += U_tensor[k, p, a, e] *
                       conj(U_tensor[m, p, b, f]) *
                       U_tensor[m, q, c, g] *
                       conj(U_tensor[k, q, d, h])
            end
            K[a, b, c, d, e, f, g, h] = val
        end
    end

    return Renyi2ContractionKernel(K)
end

"""
    get_or_build_renyi2_kernel_cache(cache::Union{TwoQubitCliffordCache, Nothing}=nothing)
        -> Vector{Renyi2ContractionKernel}

Get or build the global cache of Equation 19 contraction kernels.

This cache is built once and reused for all OBD evaluations.
Thread-safe via lock protection.

# Arguments
- `cache::Union{TwoQubitCliffordCache, Nothing}`: Optional precomputed Clifford cache

# Returns
- `Vector{Renyi2ContractionKernel}`: Vector of 11,520 contraction kernels
"""
function get_or_build_renyi2_kernel_cache(cache::Union{TwoQubitCliffordCache, Nothing}=nothing)::Vector{Renyi2ContractionKernel}
    lock(RENYI2_KERNEL_CACHE_LOCK) do
        if RENYI2_KERNEL_CACHE[] !== nothing
            return RENYI2_KERNEL_CACHE[]::Vector{Renyi2ContractionKernel}
        end

        if cache !== nothing
            matrices = cache.matrices
        else
            cliffords = get_all_two_qubit_cliffords()
            matrices = [clifford_to_matrix(C) for C in cliffords]
        end

        n_clif = length(matrices)
        kernels = Vector{Renyi2ContractionKernel}(undef, n_clif)

        for i in 1:n_clif
            kernels[i] = compute_renyi2_kernel(matrices[i])
        end

        RENYI2_KERNEL_CACHE[] = kernels
        return kernels
    end
end

"""
    evaluate_renyi2_equation19(base_tensor::Renyi2BaseTensor,
                                kernel::Renyi2ContractionKernel) -> Float64

Evaluate the Rényi-2 entropy using Equation 19 tensor contraction.

Computes Tr(ρ_n'²) = Σ T[a,b,c,d,e,f,g,h] × K[a,b,c,d,e,f,g,h]
then returns S₂ = -log(Tr(ρ_n'²)).

# Arguments
- `base_tensor::Renyi2BaseTensor`: Precomputed base tensor from MPS
- `kernel::Renyi2ContractionKernel`: Precomputed kernel for Clifford U

# Returns
- `Float64`: Second Rényi entropy S₂ = -log(Tr(ρ_n²))

# Complexity
O(256) - simple element-wise product and sum.
"""
function evaluate_renyi2_equation19(base_tensor::Renyi2BaseTensor,
                                     kernel::Renyi2ContractionKernel)::Float64
    T = base_tensor.values
    K = kernel.values

    purity = real(sum(T .* K))

    purity = clamp(purity, 1e-15, 1.0)

    return -log(purity)
end

"""
    find_optimal_clifford_equation19(base_tensor::Renyi2BaseTensor,
                                      kernels::Vector{Renyi2ContractionKernel};
                                      use_representatives::Bool=false) -> Tuple{Int, Float64}

Find the optimal Clifford to minimize Rényi-2 entropy using Equation 19.

This is the exact algorithm from Liu & Clark Section IV.A, using tensor
contraction for O(256) evaluation per Clifford.

# Arguments
- `base_tensor::Renyi2BaseTensor`: Precomputed base tensor from MPS
- `kernels::Vector{Renyi2ContractionKernel}`: Precomputed kernels for all Cliffords
- `use_representatives::Bool`: If true, only search over representative subset

# Returns
- `Tuple{Int, Float64}`: (best_clifford_index, best_entropy)

# Complexity
O(11520 × 256) = O(2.9M) for full search, which is O(1) in terms of χ.
"""
function find_optimal_clifford_equation19(base_tensor::Renyi2BaseTensor,
                                           kernels::Vector{Renyi2ContractionKernel};
                                           use_representatives::Bool=false)::Tuple{Int, Float64}
    n_search = use_representatives ? min(length(kernels), 20) : length(kernels)

    best_idx = 1
    best_entropy = Inf

    for i in 1:n_search
        entropy = evaluate_renyi2_equation19(base_tensor, kernels[i])
        if entropy < best_entropy
            best_entropy = entropy
            best_idx = i
        end
    end

    return (best_idx, best_entropy)
end

"""
    OBDAlgorithm

Algorithm selection for OBD optimization.

- `:pauli` - Pauli-basis fast evaluation (default, O(4) per Clifford)
- `:equation19` - Exact tensor contraction from Liu & Clark Eq. 19 (O(256) per Clifford)
- `:naive` - Direct matrix computation (O(128) per Clifford)
"""
const OBDAlgorithm = Symbol

"""
    find_optimal_clifford_for_bond(mps::MPS, bond::Int, sites::AbstractVector;
                                    use_full_search::Bool=false,
                                    cache::Union{TwoQubitCliffordCache, Nothing}=nothing,
                                    use_fast_algorithm::Bool=true,
                                    algorithm::Symbol=:pauli)
        -> Tuple{Int, Float64, Float64}

Find the optimal two-qubit Clifford to minimize entanglement at a bond.

# Arguments
- `mps::MPS`: Matrix Product State
- `bond::Int`: Bond index (between sites bond and bond+1)
- `sites::AbstractVector`: Site indices
- `use_full_search::Bool`: If true, search all 11,520 Cliffords; else use representatives
- `cache::Union{TwoQubitCliffordCache, Nothing}`: Precomputed Clifford cache
- `use_fast_algorithm::Bool`: If true, use fast algorithm (default); deprecated, use `algorithm`
- `algorithm::Symbol`: Algorithm to use:
  - `:pauli` - Pauli-basis fast evaluation (O(4) per Clifford, default)
  - `:equation19` - Exact tensor contraction from Eq. 19 (O(256) per Clifford)
  - `:naive` - Direct matrix computation (O(128) per Clifford)

# Returns
- `Tuple{Int, Float64, Float64}`: (best_index, initial_entropy, final_entropy)

# Algorithm Options

## Pauli-basis (default, `:pauli`)
Uses Pauli expectation values for O(4) per-Clifford evaluation:
1. Extract two-site RDM ρ (O(χ³))
2. Compute 16 Pauli expectations ⟨σ₁⊗σ₂⟩ (O(256))
3. For each Clifford, use cached Pauli mapping (O(4) per Clifford)
Complexity: O(χ³ + 11520×4) ≈ O(χ³ + 46K)

## Equation 19 (`:equation19`)
Exact tensor contraction from Liu & Clark Section IV.A:
1. Compute 256-element base tensor from MPS (O(χ³))
2. For each Clifford, contract with precomputed kernel (O(256) per Clifford)
Complexity: O(χ³ + 11520×256) ≈ O(χ³ + 2.9M)

## Naive (`:naive`)
Direct matrix computation:
1. Extract two-site RDM ρ (O(χ³))
2. For each Clifford, transform ρ and compute entropy (O(128) per Clifford)
Complexity: O(χ³ + 11520×128) ≈ O(χ³ + 1.5M)

# Note
Uses second Rényi entropy for efficiency: S_2(ρ) = -log(Tr(ρ²))
"""
function find_optimal_clifford_for_bond(mps::MPS, bond::Int, sites::AbstractVector;
                                         use_full_search::Bool=false,
                                         cache::Union{TwoQubitCliffordCache, Nothing}=nothing,
                                         use_fast_algorithm::Bool=true,
                                         algorithm::Symbol=:pauli)
    n = length(mps)
    (bond < 1 || bond >= n) && throw(ArgumentError("Invalid bond index"))

    site1 = bond
    site2 = bond + 1

    effective_algorithm = use_fast_algorithm ? algorithm : :naive


    if effective_algorithm == :equation19 && use_full_search
        base_tensor = precompute_renyi2_base_tensor(mps, bond, sites)

        kernels = get_or_build_renyi2_kernel_cache(cache)

        best_index, best_entropy = find_optimal_clifford_equation19(base_tensor, kernels;
                                                                     use_representatives=false)

        rho = extract_two_site_rdm(mps, site1, site2)
        rho_1 = partial_trace_4x4(rho, true)
        initial_entropy = compute_renyi2_entropy(rho_1)

        return (best_index, initial_entropy, best_entropy)
    end


    rho = extract_two_site_rdm(mps, site1, site2)

    rho_1 = partial_trace_4x4(rho, true)
    initial_entropy = compute_renyi2_entropy(rho_1)

    if effective_algorithm == :pauli && use_full_search
        expectations = precompute_pauli_expectations(rho)

        pauli_maps = get_or_build_clifford_pauli_cache(cache)

        best_index, best_entropy = find_optimal_clifford_fast(expectations, pauli_maps;
                                                               use_representatives=false)

        return (best_index, initial_entropy, best_entropy)
    end

    if use_full_search
        if cache !== nothing
            cliffords_matrices = cache.matrices
        else
            cliffords = get_all_two_qubit_cliffords()
            cliffords_matrices = [clifford_to_matrix(C) for C in cliffords]
        end
    else
        representatives = get_cnot_class_representatives()
        cliffords_matrices = [clifford_to_matrix(C) for C in representatives]
    end

    best_index = 1
    best_entropy = initial_entropy

    for (i, U) in enumerate(cliffords_matrices)
        rho_transformed = transform_rdm(rho, U)

        rho_1_transformed = partial_trace_4x4(rho_transformed, true)

        entropy = compute_renyi2_entropy(rho_1_transformed)

        if entropy < best_entropy
            best_entropy = entropy
            best_index = i
        end
    end

    return (best_index, initial_entropy, best_entropy)
end

"""
    apply_clifford_to_mps!(mps::MPS, C::CliffordOperator, site1::Int, site2::Int,
                            sites::AbstractVector;
                            max_bond::Int=1024, cutoff::Float64=1e-15) -> MPS

Apply a two-qubit Clifford operator to an MPS at adjacent sites.

# Arguments
- `mps::MPS`: Matrix Product State (modified in-place)
- `C::CliffordOperator`: Two-qubit Clifford to apply
- `site1::Int`: First site (must be site2 - 1)
- `site2::Int`: Second site (must be site1 + 1)
- `sites::AbstractVector`: Site indices
- `max_bond::Int`: Maximum bond dimension
- `cutoff::Float64`: SVD cutoff

# Returns
- `MPS`: Modified MPS
"""
function apply_clifford_to_mps!(mps::MPS, C::CliffordOperator, site1::Int, site2::Int,
                                 sites::AbstractVector;
                                 max_bond::Int=1024, cutoff::Float64=1e-15)::MPS
    U = clifford_to_matrix(C)
    gate = matrix_to_two_qubit_itensor(U, sites[site1], sites[site2])
    return apply_two_qubit_gate!(mps, gate, site1, site2; max_bond=max_bond, cutoff=cutoff)
end

"""
    apply_clifford_index_to_mps!(mps::MPS, clifford_index::Int, site1::Int, site2::Int,
                                  sites::AbstractVector;
                                  cache::Union{TwoQubitCliffordCache, Nothing}=nothing,
                                  use_full_search::Bool=false,
                                  max_bond::Int=1024, cutoff::Float64=1e-15) -> MPS

Apply the i-th Clifford to an MPS at adjacent sites.

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `clifford_index::Int`: Index of the Clifford
- `site1::Int`: First site
- `site2::Int`: Second site
- `sites::AbstractVector`: Site indices
- `cache::Union{TwoQubitCliffordCache, Nothing}`: Precomputed cache
- `use_full_search::Bool`: Whether full search was used (determines index interpretation)
- `max_bond::Int`: Maximum bond dimension
- `cutoff::Float64`: SVD cutoff

# Returns
- `MPS`: Modified MPS
"""
function apply_clifford_index_to_mps!(mps::MPS, clifford_index::Int, site1::Int, site2::Int,
                                       sites::AbstractVector;
                                       cache::Union{TwoQubitCliffordCache, Nothing}=nothing,
                                       use_full_search::Bool=false,
                                       max_bond::Int=1024, cutoff::Float64=1e-15)::MPS
    if use_full_search
        if cache !== nothing
            U = cache.matrices[clifford_index]
        else
            cliffords = get_all_two_qubit_cliffords()
            U = clifford_to_matrix(cliffords[clifford_index])
        end
    else
        representatives = get_cnot_class_representatives()
        U = clifford_to_matrix(representatives[clifford_index])
    end

    gate = matrix_to_two_qubit_itensor(U, sites[site1], sites[site2])
    return apply_two_qubit_gate!(mps, gate, site1, site2; max_bond=max_bond, cutoff=cutoff)
end

"""
    OBDSweepResult

Result of a single OBD sweep.

# Fields
- `initial_max_entropy::Float64`: Maximum entropy before sweep
- `final_max_entropy::Float64`: Maximum entropy after sweep
- `entropy_reduction::Float64`: Reduction in max entropy
- `applied_cliffords::Vector{Tuple{Int, Int, Int}}`: (bond, clifford_index, site1) for each applied Clifford
- `bond_entropies::Vector{Float64}`: Entropy at each bond after sweep
"""
struct OBDSweepResult
    initial_max_entropy::Float64
    final_max_entropy::Float64
    entropy_reduction::Float64
    applied_cliffords::Vector{Tuple{Int, Int, Int}}
    bond_entropies::Vector{Float64}
end

"""
    obd_sweep!(mps::MPS, sites::AbstractVector, clifford::Destabilizer;
               use_full_search::Bool=false,
               cache::Union{TwoQubitCliffordCache, Nothing}=nothing,
               max_bond::Int=1024, cutoff::Float64=1e-15,
               direction::Symbol=:left_to_right) -> Tuple{OBDSweepResult, Destabilizer}

Perform a single OBD sweep over all bonds.

# Arguments
- `mps::MPS`: Matrix Product State (modified)
- `sites::AbstractVector`: Site indices
- `clifford::Destabilizer`: Accumulated Clifford operator (modified)
- `use_full_search::Bool`: Search all 11,520 vs representatives only
- `cache::Union{TwoQubitCliffordCache, Nothing}`: Precomputed Clifford cache
- `max_bond::Int`: Maximum bond dimension
- `cutoff::Float64`: SVD cutoff
- `direction::Symbol`: :left_to_right or :right_to_left

# Returns
- `Tuple{OBDSweepResult, Destabilizer}`: (sweep result, updated Clifford)

# Algorithm
Sweep through bonds, optimizing each one:
1. Find optimal Clifford U for each bond
2. Apply U to MPS: |ψ⟩ → U|ψ⟩
3. Update Clifford: C → C·U† (so that U†·C·|ψ⟩ = C'·|ψ'⟩)
"""
function obd_sweep!(mps::MPS, sites::AbstractVector, clifford::Destabilizer;
                    use_full_search::Bool=false,
                    cache::Union{TwoQubitCliffordCache, Nothing}=nothing,
                    max_bond::Int=1024, cutoff::Float64=1e-15,
                    direction::Symbol=:left_to_right)::Tuple{OBDSweepResult, Destabilizer}

    n = length(mps)
    n >= 2 || throw(ArgumentError("MPS must have at least 2 sites"))

    initial_entropies = Float64[entanglement_entropy(mps, bond) for bond in 1:(n-1)]
    initial_max = maximum(initial_entropies)

    applied = Tuple{Int, Int, Int}[]

    if use_full_search
        all_cliffords = cache !== nothing ? cache.cliffords : get_all_two_qubit_cliffords()
    else
        all_cliffords = get_cnot_class_representatives()
    end

    bond_order = if direction == :left_to_right
        1:(n-1)
    else
        (n-1):-1:1
    end

    for bond in bond_order
        site1 = bond
        site2 = bond + 1

        best_idx, _, _ = find_optimal_clifford_for_bond(mps, bond, sites;
                                                         use_full_search=use_full_search,
                                                         cache=cache)

        if best_idx == 1
            continue
        end

        push!(applied, (bond, best_idx, site1))

        apply_clifford_index_to_mps!(mps, best_idx, site1, site2, sites;
                                      cache=cache, use_full_search=use_full_search,
                                      max_bond=max_bond, cutoff=cutoff)

        U_clifford = all_cliffords[best_idx]
        apply_clifford_to_destabilizer!(clifford, U_clifford, site1, site2; inverse=false)
    end

    final_entropies = Float64[entanglement_entropy(mps, bond) for bond in 1:(n-1)]
    final_max = maximum(final_entropies)

    result = OBDSweepResult(
        initial_max,
        final_max,
        initial_max - final_max,
        applied,
        final_entropies
    )

    return (result, clifford)
end

"""
    apply_clifford_to_destabilizer!(D::Destabilizer, C::CliffordOperator,
                                     site1::Int, site2::Int; inverse::Bool=false)

Apply a two-qubit Clifford via RIGHT MULTIPLICATION to the accumulated Destabilizer.

When U is applied to the MPS (|mps'⟩ = U|mps⟩), the Clifford must be updated
as D_new = D · U† to preserve the physical state:
    D_new |mps'⟩ = D · U† · U |mps⟩ = D |mps⟩

# Arguments
- `D::Destabilizer`: Accumulated Clifford (modified via right multiplication)
- `C::CliffordOperator`: Two-qubit Clifford that was applied to MPS
- `site1::Int`: First qubit (in the full n-qubit system)
- `site2::Int`: Second qubit
- `inverse::Bool`: If true, right-multiply by C; if false, right-multiply by C†

# Note
This embeds the 2-qubit Clifford into the n-qubit system by only acting
on the specified qubits. Uses apply_inverse_gates! from clifford_interface.jl.
"""
function apply_clifford_to_destabilizer!(D::Destabilizer, C::CliffordOperator,
                                          site1::Int, site2::Int; inverse::Bool=false)
    gates = decompose_two_qubit_clifford(C, site1, site2)

    if inverse
        inverse_gates = [inv(g) for g in reverse(gates)]
        apply_inverse_gates!(D, inverse_gates)
    else
        apply_inverse_gates!(D, gates)
    end

    return D
end

"""
    apply_clifford_left_multiply!(D::Destabilizer, C::CliffordOperator,
                                   site1::Int, site2::Int)

Apply a two-qubit Clifford via LEFT MULTIPLICATION to build up a Clifford circuit.

For building Clifford circuits: C_new = U · C_old

# Arguments
- `D::Destabilizer`: Accumulated Clifford (modified via left multiplication)
- `C::CliffordOperator`: Two-qubit Clifford to apply
- `site1::Int`: First qubit
- `site2::Int`: Second qubit
"""
function apply_clifford_left_multiply!(D::Destabilizer, C::CliffordOperator,
                                       site1::Int, site2::Int)
    gates = decompose_two_qubit_clifford(C, site1, site2)

    for gate in gates
        apply!(D, gate)
    end

    return D
end

"""
    decompose_two_qubit_clifford(C::CliffordOperator, q1::Int, q2::Int) -> Vector

Decompose a two-qubit Clifford into a sequence of elementary gates on qubits q1, q2.

# Arguments
- `C::CliffordOperator`: Two-qubit Clifford
- `q1::Int`: First qubit index in larger system
- `q2::Int`: Second qubit index in larger system

# Returns
- `Vector`: Sequence of QuantumClifford symbolic gates

# Algorithm
Uses the canonical decomposition:
    C = (L₁ ⊗ L₂) · E · (R₁ ⊗ R₂)
where L₁, L₂, R₁, R₂ are single-qubit Cliffords and E is an entangling gate.

Uses stabilizer tableau comparison for exact matching.
"""
function decompose_two_qubit_clifford(C::CliffordOperator, q1::Int, q2::Int)::Vector
    if haskey(TWO_QUBIT_CLIFFORD_DECOMPOSITION_CACHE, C)
        return remap_decomposition(TWO_QUBIT_CLIFFORD_DECOMPOSITION_CACHE[C], q1, q2)
    end

    result = find_clifford_decomposition(C, q1, q2)

    if result !== nothing
        return result
    end

    return synthesize_clifford_from_matrix(C, q1, q2)
end

"""
    find_clifford_decomposition(C::CliffordOperator, q1::Int, q2::Int) -> Union{Vector, Nothing}

Find a gate decomposition for a two-qubit Clifford by comparing stabilizer tableaux.
"""
function find_clifford_decomposition(C::CliffordOperator, q1::Int, q2::Int)::Union{Vector, Nothing}
    entangling_classes = [
        [],
        [sCNOT(1, 2)],
        [sCNOT(2, 1)],
        [sCPHASE(1, 2)],
        [sSWAP(1, 2)],
        [sCNOT(1, 2), sCNOT(2, 1)],
        [sCNOT(2, 1), sCNOT(1, 2)],
        [sCNOT(1, 2), sCNOT(2, 1), sCNOT(1, 2)],
    ]

    for entangling in entangling_classes
        result = search_with_entangling_class(C, entangling, q1, q2)
        if result !== nothing
            return result
        end
    end

    return nothing
end

"""
    search_with_entangling_class(C::CliffordOperator, entangling::Vector, q1::Int, q2::Int)

Search for single-qubit Cliffords that complete the decomposition.
"""
function search_with_entangling_class(C::CliffordOperator, entangling::Vector, q1::Int, q2::Int)::Union{Vector, Nothing}
    E_dest = one(Destabilizer, 2)
    for g in entangling
        apply!(E_dest, g)
    end
    E = CliffordOperator(E_dest)

    for r1_idx in 1:24
        for r2_idx in 1:24
            R_dest = one(Destabilizer, 2)
            for g in resolve_single_qubit_clifford_local(r1_idx, 1)
                apply!(R_dest, g)
            end
            for g in resolve_single_qubit_clifford_local(r2_idx, 2)
                apply!(R_dest, g)
            end
            R = CliffordOperator(R_dest)

            ER_dest = one(Destabilizer, 2)
            apply!(ER_dest, R)
            apply!(ER_dest, E)
            ER = CliffordOperator(ER_dest)

            ER_inv = inv(ER)

            target_L_dest = one(Destabilizer, 2)
            apply!(target_L_dest, ER_inv)
            apply!(target_L_dest, C)
            target_L = CliffordOperator(target_L_dest)

            l1_idx, l2_idx = decompose_as_tensor_product(target_L)

            if l1_idx !== nothing && l2_idx !== nothing
                gates = Vector{Any}()

                append!(gates, resolve_single_qubit_clifford_local(r1_idx, 1))
                append!(gates, resolve_single_qubit_clifford_local(r2_idx, 2))

                append!(gates, entangling)

                append!(gates, resolve_single_qubit_clifford_local(l1_idx, 1))
                append!(gates, resolve_single_qubit_clifford_local(l2_idx, 2))

                D_verify = one(Destabilizer, 2)
                for g in gates
                    apply!(D_verify, g)
                end
                C_verify = CliffordOperator(D_verify)

                if cliffords_equal(C, C_verify)
                    return remap_gates(gates, q1, q2)
                end
            end
        end
    end

    return nothing
end

"""
    decompose_as_tensor_product(C::CliffordOperator) -> Tuple{Union{Int, Nothing}, Union{Int, Nothing}}

Check if a 2-qubit Clifford is a tensor product of single-qubit Cliffords.
Returns (l1_idx, l2_idx) if successful, (nothing, nothing) otherwise.
"""
function decompose_as_tensor_product(C::CliffordOperator)::Tuple{Union{Int, Nothing}, Union{Int, Nothing}}
    for l1_idx in 1:24
        for l2_idx in 1:24
            L_dest = one(Destabilizer, 2)
            for g in resolve_single_qubit_clifford_local(l1_idx, 1)
                apply!(L_dest, g)
            end
            for g in resolve_single_qubit_clifford_local(l2_idx, 2)
                apply!(L_dest, g)
            end
            L = CliffordOperator(L_dest)

            if cliffords_equal(C, L)
                return (l1_idx, l2_idx)
            end
        end
    end

    return (nothing, nothing)
end

"""
    cliffords_equal(C1::CliffordOperator, C2::CliffordOperator) -> Bool

Check if two Clifford operators are equal by comparing their matrix representations.
Two Cliffords are equal if their matrices are equal up to global phase.
"""
function cliffords_equal(C1::CliffordOperator, C2::CliffordOperator)::Bool
    n = nqubits(C1)
    nqubits(C2) == n || return false

    U1 = clifford_to_matrix(C1)
    U2 = clifford_to_matrix(C2)

    return is_equivalent_up_to_phase(U1, U2)
end

"""
    make_single_x(n::Int, q::Int) -> PauliOperator

Create X on qubit q in an n-qubit system.
"""
function make_single_x(n::Int, q::Int)::PauliOperator
    return make_single_pauli(n, q, :X)
end

"""
    make_single_z(n::Int, q::Int) -> PauliOperator

Create Z on qubit q in an n-qubit system.
"""
function make_single_z(n::Int, q::Int)::PauliOperator
    return make_single_pauli(n, q, :Z)
end

"""
    make_single_pauli(n::Int, q::Int, p::Symbol) -> PauliOperator

Create a single-qubit Pauli operator in an n-qubit system.
"""
function make_single_pauli(n::Int, q::Int, p::Symbol)::PauliOperator
    xs = falses(n)
    zs = falses(n)

    if p == :X
        xs[q] = true
    elseif p == :Y
        xs[q] = true
        zs[q] = true
    elseif p == :Z
        zs[q] = true
    end

    return PauliOperator(0x00, xs, zs)
end

"""
    resolve_single_qubit_clifford_local(index::Int, qubit::Int) -> Vector

Resolve single-qubit Clifford to gates on local qubit index (1 or 2).
"""
function resolve_single_qubit_clifford_local(index::Int, qubit::Int)::Vector
    specs = generate_single_qubit_clifford(index, qubit)
    return [resolve_symbolic_gate(spec) for spec in specs]
end

"""
    remap_gates(gates::Vector, q1::Int, q2::Int) -> Vector

Remap gates from local indices (1, 2) to actual qubit indices (q1, q2).
"""
function remap_gates(gates::Vector, q1::Int, q2::Int)::Vector
    result = []

    for g in gates
        push!(result, remap_gate(g, q1, q2))
    end

    return result
end

"""
    remap_gate(g, q1::Int, q2::Int)

Remap a single gate from local indices to actual indices.
"""
function remap_gate(g, q1::Int, q2::Int)
    if g isa typeof(sHadamard(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sHadamard(new_target)
    elseif g isa typeof(sPhase(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sPhase(new_target)
    elseif g isa typeof(sCNOT(1, 2))
        ctrl = g.q1
        targ = g.q2
        new_ctrl = ctrl == 1 ? q1 : q2
        new_targ = targ == 1 ? q1 : q2
        return sCNOT(new_ctrl, new_targ)
    elseif g isa typeof(sCPHASE(1, 2))
        q1_local = g.q1
        q2_local = g.q2
        new_q1 = q1_local == 1 ? q1 : q2
        new_q2 = q2_local == 1 ? q1 : q2
        return sCPHASE(new_q1, new_q2)
    elseif g isa typeof(sSWAP(1, 2))
        return sSWAP(q1, q2)
    elseif g isa typeof(sX(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sX(new_target)
    elseif g isa typeof(sY(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sY(new_target)
    elseif g isa typeof(sZ(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sZ(new_target)
    elseif g isa typeof(sInvPhase(1))
        target = g.q
        new_target = target == 1 ? q1 : q2
        return sInvPhase(new_target)
    else
        @warn "Unknown gate type in remap: $(typeof(g)), returning as-is"
        return g
    end
end

"""
    remap_decomposition(decomp::Vector, q1::Int, q2::Int) -> Vector

Remap a cached decomposition to the actual qubit indices.
"""
function remap_decomposition(decomp::Vector, q1::Int, q2::Int)::Vector
    return remap_gates(decomp, q1, q2)
end

"""
    synthesize_clifford_from_matrix(C::CliffordOperator, q1::Int, q2::Int) -> Vector

Synthesize a Clifford decomposition using matrix-based approach as fallback.
"""
function synthesize_clifford_from_matrix(C::CliffordOperator, q1::Int, q2::Int)::Vector
    U = clifford_to_matrix(C)

    I4 = Matrix{ComplexF64}(LinearAlgebra.I, 4, 4)

    if is_equivalent_up_to_phase(U, I4)
        return []
    end

    CNOT_12 = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0]
    if is_equivalent_up_to_phase(U, CNOT_12)
        return [sCNOT(q1, q2)]
    end

    CNOT_21 = ComplexF64[1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]
    if is_equivalent_up_to_phase(U, CNOT_21)
        return [sCNOT(q2, q1)]
    end

    CZ_mat = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1]
    if is_equivalent_up_to_phase(U, CZ_mat)
        return [sCPHASE(q1, q2)]
    end

    SWAP_mat = ComplexF64[1 0 0 0; 0 0 1 0; 0 1 0 0; 0 0 0 1]
    if is_equivalent_up_to_phase(U, SWAP_mat)
        return [sSWAP(q1, q2)]
    end

    return brute_force_decomposition(C, q1, q2)
end

"""
    is_equivalent_up_to_phase(U1::Matrix, U2::Matrix) -> Bool

Check if two unitary matrices are equal up to a global phase.
"""
function is_equivalent_up_to_phase(U1::Matrix, U2::Matrix)::Bool
    phase_idx = findfirst(x -> abs(x) > 1e-10, U1)
    if phase_idx === nothing
        return all(abs.(U2) .< 1e-10)
    end

    if abs(U2[phase_idx]) < 1e-10
        return false
    end

    phase = U1[phase_idx] / U2[phase_idx]

    return isapprox(U1, phase * U2, atol=1e-10)
end

"""
    brute_force_decomposition(C::CliffordOperator, q1::Int, q2::Int) -> Vector

Brute-force search over all canonical decompositions.
Guaranteed to find a decomposition for any valid 2-qubit Clifford.
"""
function brute_force_decomposition(C::CliffordOperator, q1::Int, q2::Int)::Vector
    entangling_classes = [
        [],
        [sCNOT(1, 2)],
        [sCNOT(2, 1)],
        [sCPHASE(1, 2)],
        [sSWAP(1, 2)],
        [sCNOT(1, 2), sCNOT(2, 1)],
        [sCNOT(2, 1), sCNOT(1, 2)],
        [sCNOT(1, 2), sCNOT(2, 1), sCNOT(1, 2)],
        [sCNOT(2, 1), sCNOT(1, 2), sCNOT(2, 1)],
    ]

    for entangling in entangling_classes
        for l1_idx in 1:24
            for l2_idx in 1:24
                for r1_idx in 1:24
                    for r2_idx in 1:24
                        cand_dest = one(Destabilizer, 2)

                        for g in resolve_single_qubit_clifford_local(r1_idx, 1)
                            apply!(cand_dest, g)
                        end
                        for g in resolve_single_qubit_clifford_local(r2_idx, 2)
                            apply!(cand_dest, g)
                        end

                        for g in entangling
                            apply!(cand_dest, g)
                        end

                        for g in resolve_single_qubit_clifford_local(l1_idx, 1)
                            apply!(cand_dest, g)
                        end
                        for g in resolve_single_qubit_clifford_local(l2_idx, 2)
                            apply!(cand_dest, g)
                        end

                        cand = CliffordOperator(cand_dest)

                        if cliffords_equal(C, cand)
                            gates = Vector{Any}()
                            append!(gates, resolve_single_qubit_clifford_local(r1_idx, 1))
                            append!(gates, resolve_single_qubit_clifford_local(r2_idx, 2))
                            append!(gates, entangling)
                            append!(gates, resolve_single_qubit_clifford_local(l1_idx, 1))
                            append!(gates, resolve_single_qubit_clifford_local(l2_idx, 2))

                            return remap_gates(gates, q1, q2)
                        end
                    end
                end
            end
        end
    end

    @warn "Could not decompose Clifford - this may indicate a bug"
    return []
end

const TWO_QUBIT_CLIFFORD_DECOMPOSITION_CACHE = Dict{CliffordOperator, Vector}()

"""
    OBDResult

Result of full OBD optimization.

# Fields
- `num_sweeps::Int`: Number of sweeps performed
- `converged::Bool`: Whether optimization converged
- `initial_max_entropy::Float64`: Initial maximum bond entropy
- `final_max_entropy::Float64`: Final maximum bond entropy
- `sweep_results::Vector{OBDSweepResult}`: Results from each sweep
- `total_cliffords_applied::Int`: Total number of Cliffords applied
"""
struct OBDResult
    num_sweeps::Int
    converged::Bool
    initial_max_entropy::Float64
    final_max_entropy::Float64
    sweep_results::Vector{OBDSweepResult}
    total_cliffords_applied::Int
end

"""
    obd!(state::CAMPSState; max_sweeps::Int=10,
         improvement_threshold::Float64=1e-10,
         use_full_search::Bool=false,
         cache::Union{TwoQubitCliffordCache, Nothing}=nothing) -> OBDResult

Apply full OBD optimization to a CAMPS state.

This is the main OBD entry point for reducing entanglement in the MPS.

# Arguments
- `state::CAMPSState`: CAMPS state (modified in-place)
- `max_sweeps::Int`: Maximum number of bidirectional sweeps
- `improvement_threshold::Float64`: Stop if improvement < this
- `use_full_search::Bool`: Search all 11,520 Cliffords vs representatives
- `cache::Union{TwoQubitCliffordCache, Nothing}`: Precomputed cache

# Returns
- `OBDResult`: Optimization results

# Algorithm
1. Perform left-to-right sweep
2. Perform right-to-left sweep
3. Repeat until convergence or max_sweeps reached

# Example
```julia
state = CAMPSState(10)
initialize!(state)
# ... apply some gates that increase entanglement ...

result = obd!(state; max_sweeps=5)
println("Reduced max entropy from \$(result.initial_max_entropy) to \$(result.final_max_entropy)")
```
"""
function obd!(state::CAMPSState; max_sweeps::Int=10,
              improvement_threshold::Float64=1e-10,
              use_full_search::Bool=false,
              cache::Union{TwoQubitCliffordCache, Nothing}=nothing)::OBDResult

    ensure_initialized!(state)

    n = state.n_qubits
    if n < 2
        return OBDResult(0, true, 0.0, 0.0, OBDSweepResult[], 0)
    end

    sweep_results = OBDSweepResult[]
    initial_max = max_entanglement_entropy(state.mps)
    total_cliffords = 0
    converged = false

    for sweep_num in 1:max_sweeps
        result_lr, _ = obd_sweep!(state.mps, state.sites, state.clifford;
                                   use_full_search=use_full_search,
                                   cache=cache,
                                   max_bond=state.max_bond,
                                   cutoff=state.cutoff,
                                   direction=:left_to_right)
        push!(sweep_results, result_lr)
        total_cliffords += length(result_lr.applied_cliffords)

        result_rl, _ = obd_sweep!(state.mps, state.sites, state.clifford;
                                   use_full_search=use_full_search,
                                   cache=cache,
                                   max_bond=state.max_bond,
                                   cutoff=state.cutoff,
                                   direction=:right_to_left)
        push!(sweep_results, result_rl)
        total_cliffords += length(result_rl.applied_cliffords)

        improvement = result_lr.entropy_reduction + result_rl.entropy_reduction
        if improvement < improvement_threshold
            converged = true
            break
        end
    end

    final_max = max_entanglement_entropy(state.mps)

    return OBDResult(
        length(sweep_results) ÷ 2,
        converged,
        initial_max,
        final_max,
        sweep_results,
        total_cliffords
    )
end

"""
    apply_rotation_with_obd!(state::CAMPSState, P_twisted::PauliOperator, θ::Real;
                              obd_sweeps::Int=2,
                              use_full_search::Bool=false) -> CAMPSState

Apply a non-Clifford rotation using OBD for entanglement reduction.

This is the fallback when OFD cannot be applied (no free qubit with X/Y).

# Arguments
- `state::CAMPSState`: CAMPS state (modified)
- `P_twisted::PauliOperator`: Twisted Pauli operator
- `θ::Real`: Rotation angle
- `obd_sweeps::Int`: Number of OBD sweeps after rotation
- `use_full_search::Bool`: Use full Clifford search

# Returns
- `CAMPSState`: Modified state

# Algorithm
1. Apply the twisted rotation directly to MPS (may increase bond dimension)
2. Run OBD to reduce entanglement
3. Record the twisted Pauli for GF(2) tracking
"""
function apply_rotation_with_obd!(state::CAMPSState, P_twisted::PauliOperator, θ::Real;
                                   obd_sweeps::Int=2,
                                   use_full_search::Bool=false)::CAMPSState
    ensure_initialized!(state)

    apply_twisted_rotation!(state.mps, state.sites, P_twisted, Float64(θ);
                            max_bond=state.max_bond, cutoff=state.cutoff)

    add_twisted_pauli!(state, P_twisted)

    if obd_sweeps > 0
        obd!(state; max_sweeps=obd_sweeps, use_full_search=use_full_search)
    end

    return state
end

"""
    apply_rotation_hybrid!(state::CAMPSState, axis::Symbol, qubit::Int, θ::Real;
                            strategy::DisentanglingStrategy=HybridStrategy()) -> CAMPSState

Apply a non-Clifford rotation using the specified disentangling strategy.

This is the main entry point for applying non-Clifford gates with disentangling.

# Arguments
- `state::CAMPSState`: CAMPS state (modified)
- `axis::Symbol`: Rotation axis (:X, :Y, or :Z)
- `qubit::Int`: Target qubit
- `θ::Real`: Rotation angle
- `strategy::DisentanglingStrategy`: OFD, OBD, Hybrid, or NoDisentangling

# Returns
- `CAMPSState`: Modified state

# Example
```julia
state = CAMPSState(5)
initialize!(state)
apply_clifford_gate!(state.clifford, sHadamard(1))

# Apply T gate with hybrid strategy
apply_rotation_hybrid!(state, :Z, 1, π/4)
```
"""
function apply_rotation_hybrid!(state::CAMPSState, axis::Symbol, qubit::Int, θ::Real;
                                 strategy::DisentanglingStrategy=HybridStrategy())::CAMPSState
    ensure_initialized!(state)

    P_twisted = compute_twisted_pauli(state, axis, qubit)

    if strategy isa OFDStrategy
        success, _ = try_apply_ofd!(state, P_twisted, θ)
        if !success
            @debug "OFD failed for rotation on qubit $qubit, applying directly"
            apply_twisted_rotation!(state.mps, state.sites, P_twisted, Float64(θ);
                                    max_bond=state.max_bond, cutoff=state.cutoff)
            add_twisted_pauli!(state, P_twisted)
        end

    elseif strategy isa OBDStrategy
        apply_rotation_with_obd!(state, P_twisted, θ;
                                  obd_sweeps=strategy.max_sweeps,
                                  use_full_search=false)

    elseif strategy isa HybridStrategy
        success, _ = try_apply_ofd!(state, P_twisted, θ)
        if !success
            apply_rotation_with_obd!(state, P_twisted, θ;
                                      obd_sweeps=strategy.obd_sweeps_on_failure,
                                      use_full_search=false)
        end

    elseif strategy isa NoDisentangling
        apply_twisted_rotation!(state.mps, state.sites, P_twisted, Float64(θ);
                                max_bond=state.max_bond, cutoff=state.cutoff)
        add_twisted_pauli!(state, P_twisted)

    else
        throw(ArgumentError("Unknown strategy type: $(typeof(strategy))"))
    end

    return state
end

"""
    apply_t_gate_hybrid!(state::CAMPSState, qubit::Int;
                          strategy::DisentanglingStrategy=HybridStrategy()) -> CAMPSState

Apply a T gate using the specified disentangling strategy.

Convenience function for T gates (the most common non-Clifford gate).

# Arguments
- `state::CAMPSState`: CAMPS state
- `qubit::Int`: Target qubit
- `strategy::DisentanglingStrategy`: Disentangling strategy

# Returns
- `CAMPSState`: Modified state
"""
function apply_t_gate_hybrid!(state::CAMPSState, qubit::Int;
                               strategy::DisentanglingStrategy=HybridStrategy())::CAMPSState
    return apply_rotation_hybrid!(state, :Z, qubit, π/4; strategy=strategy)
end

"""
    apply_tdag_gate_hybrid!(state::CAMPSState, qubit::Int;
                             strategy::DisentanglingStrategy=HybridStrategy()) -> CAMPSState

Apply a T† gate using the specified disentangling strategy.

# Arguments
- `state::CAMPSState`: CAMPS state
- `qubit::Int`: Target qubit
- `strategy::DisentanglingStrategy`: Disentangling strategy

# Returns
- `CAMPSState`: Modified state
"""
function apply_tdag_gate_hybrid!(state::CAMPSState, qubit::Int;
                                  strategy::DisentanglingStrategy=HybridStrategy())::CAMPSState
    return apply_rotation_hybrid!(state, :Z, qubit, -π/4; strategy=strategy)
end

"""
    get_entropy_profile(state::CAMPSState) -> Vector{Float64}

Get the entanglement entropy at each bond of the CAMPS state.

# Arguments
- `state::CAMPSState`: CAMPS state

# Returns
- `Vector{Float64}`: Entropy at each bond [S₁, S₂, ..., S_{n-1}]
"""
function get_entropy_profile(state::CAMPSState)::Vector{Float64}
    ensure_initialized!(state)
    return entanglement_entropy_all_bonds(state.mps)
end

"""
    get_bond_dimension_profile(state::CAMPSState) -> Vector{Int}

Get the bond dimension at each bond of the CAMPS state.

# Arguments
- `state::CAMPSState`: CAMPS state

# Returns
- `Vector{Int}`: Bond dimension at each bond [χ₁, χ₂, ..., χ_{n-1}]
"""
function get_bond_dimension_profile(state::CAMPSState)::Vector{Int}
    ensure_initialized!(state)
    return all_bond_dimensions(state.mps)
end

"""
    estimate_obd_improvement(state::CAMPSState; use_full_search::Bool=false) -> NamedTuple

Estimate how much OBD could reduce entanglement without actually applying it.

# Arguments
- `state::CAMPSState`: CAMPS state (not modified)
- `use_full_search::Bool`: Use full Clifford search

# Returns
- `NamedTuple` with fields:
  - `current_max_entropy::Float64`: Current maximum bond entropy
  - `estimated_reduction::Float64`: Estimated entropy reduction
  - `best_bond::Int`: Bond with highest potential improvement
"""
function estimate_obd_improvement(state::CAMPSState; use_full_search::Bool=false)
    ensure_initialized!(state)

    n = state.n_qubits
    if n < 2
        return (current_max_entropy=0.0, estimated_reduction=0.0, best_bond=0)
    end

    current_entropies = get_entropy_profile(state)
    current_max = maximum(current_entropies)

    best_reduction = 0.0
    best_bond = 1

    for bond in 1:(n-1)
        _, initial, final = find_optimal_clifford_for_bond(state.mps, bond, state.sites;
                                                            use_full_search=use_full_search)
        reduction = initial - final
        if reduction > best_reduction
            best_reduction = reduction
            best_bond = bond
        end
    end

    return (
        current_max_entropy = current_max,
        estimated_reduction = best_reduction,
        best_bond = best_bond
    )
end
