using BenchmarkTools
using QuadGK
using LinearAlgebra
using Plots

# Define the two double well potentials with the shift
shift = 0.5
function VP(X)
    return (X⋅X - 1)^2/4;
end
function VQ(X)
    return ((X - shift)⋅(X - shift) - 1)^2/4;
end
f = x -> Int(1 + shift - 0.05 < x[1] < 1 + shift + 0.05);

errors = []
bounds = []
rel_entropies = []
variances = []
β_vals = LinRange(1.0, 100.0, 25)
for β in β_vals
    # Set up the distributions
    ZP = quadgk(x -> exp(-β*VP(x)), -Inf, Inf)[1]
    ZQ = quadgk(x -> exp(-β*VQ(x)), -Inf, Inf)[1]
    P(x) = exp(-β*VP(x))/ZP
    Q(x) = exp(-β*VQ(x))/ZQ

    # Find the true expectations with quadrature and compute the real epistemic error
    true_expectation_P = quadgk(x -> P(x), 1 + shift - 0.05, 1 + shift + 0.05, rtol = 1e-10)[1]
    println("EP = $(true_expectation_P)")
    true_expectation_Q = quadgk(x -> Q(x), 1 + shift - 0.05, 1 + shift + 0.05, rtol = 1e-10)[1]
    println("EQ = $(true_expectation_Q)")
    append!(errors, true_expectation_Q - true_expectation_P)
    
    # Compute the relative entropy using the fact that they are both Boltzmann distributions
    relative_entropy = β*quadgk(x -> (VQ(x) - VP(x))*P(x), -Inf, Inf, rtol = 1e-10)[1] + log(ZQ/ZP)
    append!(rel_entropies, relative_entropy)

    # Compute the variance of the observable under the nominal distribution
    probability_of_indicator = quadgk(x -> P(x), 1 + shift - 0.05, 1 + shift + 0.05, rtol = 1e-10)[1]
    variance = probability_of_indicator*(1 - probability_of_indicator)
    append!(variances, variance)

    # Compute the linearized bound from the variance and the relative entropy
    linearized_bound = sqrt.(variance)*sqrt(2*relative_entropy)
    append!(bounds, linearized_bound)
    println("Done with β = $(β)")
end

# Plot the errors and the linearized bound
plot(β_vals, errors, title = "Epistemic Error (Shifted Double Well)", xlabel = "β", ylabel = "Error", label = "True Error")
plot!(β_vals, bounds, label = "Upper Bound")

