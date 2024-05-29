using BenchmarkTools
using QuadGK
using Distributions
using Plots

include("we_example_compute.jl")

ZP = quadgk(x -> exp(-β*VP(x)), -Inf, Inf)[1]
ZQ = quadgk(x -> exp(-β*VQ(x)), -Inf, Inf)[1]

P(x) = exp(-β*VP(x))/ZP
Q(x) = exp(-β*VQ(x))/ZQ

relative_entropy = β*quadgk(x -> (VQ(x) - VP(x))*P(x), -Inf, Inf, rtol = 1e-10)[1] + log(ZQ/ZP)

EPf = quadgk(x -> P(x), 1 + shift - 0.05, 1 + shift + 0.05)[1]

# Compute the true expectation with quadrature
c_vals = LinRange(0.1, 5, 50)
theta_vals = []
for c in c_vals
    println("c = $(c)")
    observable(x) = exp(c*(f(x) - EPf))
    true_expectation_P = quadgk(x -> observable(x)*P(x), -Inf, Inf, rtol = 1e-10)[1]
    println("True expectation under P: ", true_expectation_P)

    # Compute the expectation using the result of the WE simulation
    estimated_expectation = mean([observable.(E.ξ) ⋅ E.ω for E in E_trajectory])
    println("Estimated expectation under P with WE: ", estimated_expectation)
    Θ = log(estimated_expectation)/c + relative_entropy/c
    append!(theta_vals, Θ)
end

plot(c_vals, theta_vals)