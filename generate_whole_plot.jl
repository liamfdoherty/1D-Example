using Plots
using JLD2

errors = load("True_Errors.jld2")["values"]
linear_bounds = load("Linearized_Bounds.jld2")["values"]
UQII_MC = load("UQII_MC_NEW.jld2")["values"]
UQII_WE = load("UQII_UpTo_100.jld2")["values"]

β_vals = LinRange(1.0, 100.0, 25)

plt = plot(β_vals, errors, title = "Linearized Bound and UQIIs vs. True Error", label = "True Errors", legend = :topright)
scatter!(β_vals, linear_bounds, label = "Linearized Bounds")
scatter!(β_vals, UQII_MC, label = "MC UQII")
scatter!(β_vals, UQII_WE, label = "WE UQII")
xlabel!("β")
ylabel!("Upper Bound")

display(plt)