using LinearAlgebra
using Plots
using Random

Random.seed!(2)

function VP(X)
    return (X⋅X - 1)^2/4;
end
function gradVP!(gradVP, X)
    @. gradVP = X ⋅ (X⋅X - 1);
    gradVP
end

shift = 0.25
function VQ(X)
    return ((X - shift)⋅(X - shift) - 1)^2/4;
end

f = x -> Int(1 + shift - 0.05 < x[1] < 1 + shift + 0.05);

function metropolis_sampler(density, num_samples, initial_point = -1.0)
    samples = [initial_point]
    current_sample = initial_point
    for _ in 2:num_samples
        proposal = current_sample + 0.1*randn()
        acceptance_ratio = density(proposal)/density(current_sample)
        threshold = rand()
        if threshold <= acceptance_ratio
            current_sample = proposal
        end
        push!(samples, current_sample)
    end
    return samples
end

sample_size = Int(1e7)
theta_vectors = []
optimal_values = []
errors = []
c_vals = LinRange(0.01, 50, 50)
β_vals = LinRange(1., 100, 25)
for β in β_vals
    ZP = quadgk(x -> exp(-β*VP(x)), -Inf, Inf)[1]
    ZQ = quadgk(x -> exp(-β*VQ(x)), -Inf, Inf)[1]

    P(x) = exp(-β*VP(x))/ZP
    Q(x) = exp(-β*VQ(x))/ZQ 

    relative_entropy = β*quadgk(x -> (VQ(x) - VP(x))*P(x), -Inf, Inf, rtol = 1e-10)[1] + log(ZQ/ZP)

    true_expectation_P = quadgk(x -> P(x), 1 + shift - 0.05, 1 + shift + 0.05, rtol = 1e-10)[1]
    println("EP = $(true_expectation_P)")
    true_expectation_Q = quadgk(x -> Q(x), 1 + shift - 0.05, 1 + shift + 0.05, rtol = 1e-10)[1]
    println("EQ = $(true_expectation_Q)")
    append!(errors, true_expectation_Q - true_expectation_P)

    include("we_example_setup.jl")
    sampler = MALA(VP, gradVP!, β, Δt)
    n_experiments = 100
    sample_list = []
    for experiment in 1:n_experiments
        print("Experiment $(experiment)")
        sampling_results = []
        n_iters = 10^8
        n_save_iters = 10^1
        samples, acceptance_rates = sample_trajectory(x₀, sampler, options=MDOptions(n_iters=n_iters,n_save_iters=n_iters));
        append!(sample_list, samples)
    end
    println("Sampling Complete")
    EPf = quadgk(x -> P(x), 1 + shift - 0.05, 1 + shift + 0.05)[1]

    theta_vals = []
    B = 2*relative_entropy
    for c in c_vals
        println("c = $(c)")
        observable(x) = exp(c*(f(x) - EPf))
        true_expectation_P = quadgk(x -> observable(x)*P(x), -Inf, Inf, rtol = 1e-10)[1]
        println("True expectation under P: ", true_expectation_P)

        # Compute the expectation using the result of the WE simulation
        estimated_expectation = mean(observable.(sample_list))
        println("Estimated expectation under P with MCMC: ", estimated_expectation)
        Θ = log(estimated_expectation)/c + B/c
        append!(theta_vals, Θ)
    end
    push!(theta_vectors, theta_vals)
    # Construct the interpolant
    itp = interpolate(theta_vals, BSpline(Linear()))
    sitp = Interpolations.scale(itp, c_vals)
    extp = Interpolations.extrapolate(sitp, Interpolations.Flat())

    # Optimize the interpolant
    initial_c = 1.0
    opt = optimize(c -> extp(first(c)), [initial_c])
    print("Optimization result: $(opt)")
    optimal_value = minimum(opt)
    push!(optimal_values, optimal_value)
    println("Done β = $(β)")
end

uqii_plot = plot(β_vals, optimal_values, title = "UQII by β Value", xlabel = "β", ylabel = "Bound", label = "UQII")
plot!(β_vals, errors, label = "True Error")

