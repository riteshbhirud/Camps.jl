using Test
using JET
using CAMPS

@testset "JET" begin
    rep = JET.report_package(CAMPS; target_modules=(CAMPS,))
    println(rep)
    @test length(JET.get_reports(rep)) == 0
end
