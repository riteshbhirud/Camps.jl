using Test
using Aqua
using CAMPS

@testset "Aqua" begin
    Aqua.test_all(CAMPS;
        ambiguities=false,  # can enable later once fixed
        stale_deps=(; ignore=[:CairoMakie, :BenchmarkTools, :JSON, :Dates]),  # adjust as needed
    )
end
