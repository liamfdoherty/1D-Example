using BenchmarkTools
using QuadGK
using LinearAlgebra
using Plots

# TODO: Show that in the case with fixed temperature, but moving double well potentials,
# the linearized bounds track, and grow with the relative entropy

# Define the observable centered around the fixed alternative (hidden) distribution
left = -0.05; right = 0.05
f = x -> Int(left < x[1] < right);
β = 25.
shift_vals = LinRange(0, 2, 100)

errors = []
rel_entropies = []
variances = []
bounds = []
for shift in shift_vals
    # Define the two double well potentials with the shift
    function VP(X)
        return (X⋅X - 1)^2/4;
    end
    function VQ(X)
        return ((X - shift)⋅(X - shift) - 1)^2/4;
    end

    # Write down the normalized distributions (not optimal, since ZP and ZQ are the same at fixed temperature)
    ZP = quadgk(x -> exp(-β*VP(x)), -Inf, Inf)[1]
    ZQ = quadgk(x -> exp(-β*VQ(x)), -Inf, Inf)[1]
    P(x) = exp(-β*VP(x))/ZP
    Q(x) = exp(-β*VQ(x))/ZQ

    # Find the true expectations with quadrature and compute the real epistemic error
    true_expectation_P = quadgk(x -> P(x), left, right, rtol = 1e-10)[1]
    println("EP = $(true_expectation_P)")
    true_expectation_Q = quadgk(x -> Q(x), left, right, rtol = 1e-10)[1]
    println("EQ = $(true_expectation_Q)")
    append!(errors, true_expectation_Q - true_expectation_P)

    # Compute the relative entropy using the fact that they are both Boltzmann distributions
    relative_entropy = β*quadgk(x -> (VQ(x) - VP(x))*P(x), -Inf, Inf, rtol = 1e-10)[1] + log(ZQ/ZP)
    append!(rel_entropies, relative_entropy)

    # Compute the variance of the observable under the nominal distribution
    probability_of_indicator = quadgk(x -> P(x), left, right, rtol = 1e-10)[1]
    variance = probability_of_indicator*(1 - probability_of_indicator)
    append!(variances, variance)

    # Compute the linearized bound from the variance and the relative entropy
    linearized_bound = sqrt.(variance)*sqrt(2*relative_entropy)
    append!(bounds, linearized_bound)
    println("Done with β = $(β)")
end

# Plot the errors and the linearized bound
plot(shift_vals, errors, title = "Epistemic Error (Shifted Double Well)", xlabel = "Shift", ylabel = "Error", label = "True Error")
plot!(shift_vals, bounds, label = "Upper Bound")
plot!(shift_vals, -bounds, label = "Lower Bound")