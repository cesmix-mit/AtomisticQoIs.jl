using SafeTestsets

@safetestset "Gibbs Tests" begin include("distribution_tests.jl") end
# @safetestset "Discrepancy Tests" begin include("discrepancy_tests.jl") end
@safetestset "QoI Int. Tests" begin include("qoi_integration_tests.jl") end