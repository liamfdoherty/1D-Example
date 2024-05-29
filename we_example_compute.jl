using Statistics
using WeightedEnsemble

# include("we_example_setup.jl")

n_save_iters = 10^1
nΔt_coarse = n_save_iters
n_we_steps = nΔt ÷ n_save_iters
n_samples_per_bin = 10^4
n_particles = 10^4

# define bin structure
voronoi_pts = [[x] for x in LinRange(a-.1,b+.1,21)];
B₀, bin_id, rebin! = setup_Voronoi_bins(voronoi_pts);

# define the mutation mapping
opts = MDOptions(n_iters=nΔt_coarse, n_save_iters = nΔt_coarse)
mutation! = x-> sample_trajectory!(x, sampler_P, options=opts);

# construct coarse model matrix
# Random.seed!(1);
x0_vals = copy(voronoi_pts);
n_bins = length(B₀);
K̃ = WeightedEnsemble.build_coarse_transition_matrix(mutation!, bin_id, x0_vals, n_bins, n_samples_per_bin);

# define coarse observable as a bin function
f̃ = f.(voronoi_pts);
_,v²_vectors = WeightedEnsemble.build_coarse_vectors(n_we_steps,K̃,float.(f̃));
v² = (x,t)-> v²_vectors[t+1][bin_id(x)]

# define selection function
selection! = (E, B, t)-> optimal_selection!(E, B, v², t)

# set up ensemble
E₀ = Dirac_to_Ensemble(x₀, n_particles);
rebin!(E₀, B₀, 0);

we_sampler = WEsampler(mutation!, selection!, rebin!);

# Run WE
# Random.seed!(2);
E_trajectory, B_trajectory = run_we(E₀, B₀, we_sampler, n_we_steps);
