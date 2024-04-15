"""

AtomisticQoIs

Author: Joanna Zou <jjzou@mit.edu>
Version: 0.1.0
Year: 2024
Notes: A Julia library for computing expectation-valued quantities of interest (QoIs) from molecular dynamics simulation.

"""
module AtomisticQoIs

using LinearAlgebra
using Random
using Statistics
using StatsBase
using StatsFuns
using Distributions
using FastGaussQuadrature
using ForwardDiff
using AdvancedHMC
using PotentialLearning

# functions for MCMC sampling from distributions
include("Sampling/sampling.jl")

# functions for defining integration method and parameters
include("Integrators/integrators.jl")

# functions for defining generic distributions
include("Distributions/distributions.jl")

# functions for computing expectations (QoIs)
include("QoI/qoi.jl")

# functions for computing error/diagnostic metrics
include("Diagnostics/diagnostics.jl")


end
