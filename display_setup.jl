using Plots
using QuadGK
using LinearAlgebra

shift = 0.25
function VP(X)
    return (X⋅X - 1)^2/4;
end
function VQ(X)
    return ((X - shift)⋅(X - shift) - 1)^2/4;
end
f = x -> Int(1 + shift - 0.05 < x[1] < 1 + shift + 0.05);

β = 50.

ZP = quadgk(x -> exp(-β*VP(x)), -Inf, Inf)[1]
ZQ = quadgk(x -> exp(-β*VQ(x)), -Inf, Inf)[1]
P(x) = exp(-β*VP(x))/ZP
Q(x) = exp(-β*VQ(x))/ZQ

x_vals = LinRange(-2, 2, 2001)

plt = plot(x_vals, P.(x_vals), title = "Nominal, Alternative, Observable: β = 50, s = 0.25", label = "P", color = :red)
plot!(x_vals, Q.(x_vals), label = "Q", color = :blue)
plot!(x_vals, f.(x_vals), label = "f", color = :orange, linestyle = :dash)
display(plt)