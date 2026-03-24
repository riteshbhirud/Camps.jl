using Test
using CAMPS

@testset "CAMPS.jl" begin
    @testset "Phase 1: Foundation" begin
        include("test_types.jl")
        include("test_utils.jl")
    end

    @testset "Phase 2: Library Interfaces" begin
        include("test_clifford_interface.jl")
        include("test_mps_interface.jl")
        include("test_gf2.jl")
        include("test_gf2_circuit_analysis.jl")
    end

    @testset "Phase 2: Integration" begin
        include("test_integration.jl")
    end

    @testset "Phase 3: Core Algorithms" begin
        include("test_ofd.jl")
        include("test_obd.jl")
    end

    @testset "Phase 4: Simulation" begin
        include("test_simulation.jl")
        include("test_phase4.jl")
    end

    @testset "Phase 5: Mathematical Properties" begin
        include("test_mathematical_properties.jl")
    end

    @testset "Phase 6: End-to-End Correctness" begin
        include("test_correctness.jl")
    end

    @testset "Code Quality" begin
        include("test_aqua.jl")
        include("test_jet.jl")
    end
end
