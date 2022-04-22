include("experiment.jl")

run_experiment("data/opt_adam_4dim.h5", learning_rate=[1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                feedback_factor=[0.1, 1, 5, 10], synapseType=[AdamSynapse], striatumUpdate=[:dopamine], repetition=1:10)