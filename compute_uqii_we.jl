using Plots
using Optim
using Interpolations
using LinearAlgebra
using BasicMD
using QuadGK

β_vals = LinRange(1, 100, 25)

shift = 0.25
function VP(X)
    return (X⋅X - 1)^2/4;
end
function gradVP!(gradVP, X)
    @. gradVP = X ⋅ (X⋅X - 1);
    gradVP
end
function VQ(X)
    return ((X - shift)⋅(X - shift) - 1)^2/4;
end
f = x -> Int(1 + shift - 0.05 < x[1] < 1 + shift + 0.05);

a = -2.0; b = 2.0;
x₀ = [-1.0]; # starting point
Δt = 1e-2;  # time step
T = 1e3;    # terminal time
M = 1.0
# nΔt = ceil(Int, T/Δt);
nΔt = 100

c_vals = LinRange(0.5, 50, 50)

theta_vectors = []
optimal_values = []
errors = []
for β in β_vals
    # global sampler_P = HMC(VP, gradVP!, β, M, Δt, nΔt);
    global sampler_P = MALA(VP, gradVP!, β, Δt);
    include("we_example_compute.jl")

    ZP = quadgk(x -> exp(-β*VP(x)), -Inf, Inf)[1]
    ZQ = quadgk(x -> exp(-β*VQ(x)), -Inf, Inf)[1]

    P(x) = exp(-β*VP(x))/ZP
    Q(x) = exp(-β*VQ(x))/ZQ

    relative_entropy = β*quadgk(x -> (VQ(x) - VP(x))*P(x), -Inf, Inf, rtol = 1e-10)[1] + log(ZQ/ZP)

    EPf = quadgk(x -> P(x), 1 + shift - 0.05, 1 + shift + 0.05)[1]

    true_expectation_P = quadgk(x -> P(x), 1 + shift - 0.05, 1 + shift + 0.05, rtol = 1e-10)[1]
    println("EP = $(true_expectation_P)")
    true_expectation_Q = quadgk(x -> Q(x), 1 + shift - 0.05, 1 + shift + 0.05, rtol = 1e-10)[1]
    println("EQ = $(true_expectation_Q)")
    append!(errors, true_expectation_Q - true_expectation_P)

    theta_vals = []
    B = 2*relative_entropy
    for c in c_vals
        println("c = $(c)")
        observable(x) = exp(c*(f(x) - EPf))
        true_expectation_P = quadgk(x -> observable(x)*P(x), -Inf, Inf, rtol = 1e-10)[1]
        println("True expectation under P: ", true_expectation_P)

        # Compute the expectation using the result of the WE simulation
        estimated_expectation = mean([observable.(E.ξ) ⋅ E.ω for E in E_trajectory])
        println("Estimated expectation under P with WE: ", estimated_expectation)
        Θ = log(estimated_expectation)/c + B/c
        append!(theta_vals, Θ)
    end
    push!(theta_vectors, theta_vals)
    
    # Construct the interpolant
    itp = interpolate(theta_vals, BSpline(Linear()))
    sitp = Interpolations.scale(itp, c_vals)

    # Optimize the interpolant
    initial_c = 1.0
    opt = optimize(c -> sitp(first(c)), [initial_c])
    print("Optimization result: $(opt)")
    optimal_value = minimum(opt)
    push!(optimal_values, optimal_value)
    println("Done β = $(β)")
end

theta_plot = plot(LinRange(0.5, 20, 50), theta_vectors, label = β_vals', title = "Theta Functions to be optimized by β", legend = :outertopright)
display(theta_plot)
uqii_plot = plot(β_vals, optimal_values, title = "UQII by β Value", xlabel = "β", ylabel = "Bound", label = "UQII")
plot!(β_vals, errors, label = "True Error")