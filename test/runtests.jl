using SafeTestsets

@safetestset "Gibbs Tests" begin include("distribution_tests.jl") end
# @safetestset "Discrepancy Tests" begin include("discrepancy_tests.jl") end
@safetestset "QoI Tests" begin include("qoi_tests.jl") end