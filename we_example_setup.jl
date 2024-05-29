using LinearAlgebra
using Random
using BasicMD

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

function gradVQ!(gradVQ, X)
    @. gradVQ = (X - shift) ⋅ ((X - shift)⋅(X - shift) - 1);
    gradVQ
end

a = -2.0; b = 2.0;
x₀ = [-1.0]; # starting point
Δt = 1e-2;  # time step
β = 10.0;  # inverse temperature
T = 1e4;    # terminal time
nΔt = ceil(Int, T/Δt);
M = 1.0

f = x -> Int(1 + shift - 0.05 < x[1] < 1 + shift + 0.05); # define observable

# define sampler
sampler_P = MALA(VP, gradVP!, β, Δt);

sampler_Q = MALA(VQ, gradVQ!, β, Δt);
