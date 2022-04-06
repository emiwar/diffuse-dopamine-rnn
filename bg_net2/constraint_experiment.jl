using LinearAlgebra
using ProgressMeter
using Statistics
using Distributions
using IterTools
using HDF5
include("bg_net_v2.jl")

function run_trial(net::BgNet, target::AbstractArray, input::AbstractArray,
                   updateRule::Symbol)
    loss = 0.0
    for t=1:size(target, 1)
        loss += step!(net, target[t, :], clamp=(thal=input[t, :],),
                      updateStriatum=updateRule)
    end
    return loss
end

function create_input(thal_size, trial_length)
    proj = randn(thal_size, 2)
    proj = 4*(proj ./ sqrt.(sum(proj.^2, dims=2)))
    theta = 2*pi/trial_length * (1:trial_length)
    return phi.(proj*[cos.(theta) sin.(theta)]')'
end

const DEFAULT_PARAMS = (net_size=200, trial_length=200,
                        target_dim=4, target_tau=20.0, lambda=0.1,
                        n_trials=5000, striatumUpdate=:dopamine,
                        learning_rate=1e-3, feedback_factor=25,
                        n_varicosities=10,
                        ctx_str=:plastic, thal_str=:plastic, 
                        str_str=:plastic, str_snr=:plastic,
                        repetition=1)

function ConstrainedBgNet(size::Integer, readout_size::Integer, eta_snr::Float64,
                          eta_str::Float64; lambda=0.1, n_varicosities::Integer=10,
                          ctx_str::Symbol=:plastic, thal_str::Symbol=:plastic, 
                          str_str::Symbol=:plastic, str_snr::Symbol=:plastic)
    populations = Dict{Symbol, Population}()
    
    populations[:ctx_exc] = Population(floor(Int, 0.8*size), tau=10.0, noise=0.0)
    populations[:ctx_inh] = Population(floor(Int, 0.2*size), tau=10.0, noise=0.0)
    populations[:str_dmsn] = Population(floor(Int, 0.5*size), tau=10.0, noise=0.0)
    populations[:str_imsn] = Population(floor(Int, 0.5*size), tau=10.0, noise=0.0)
    populations[:snr] = Population(readout_size, tau=0.0, bias=1.0)
    populations[:thal] = Population(readout_size*10, tau=0.0, bias=0.0, noise=0.0)

    connect!(StaticSynapse, populations[:ctx_exc], populations[:ctx_exc],
             (pre, post)->25*rand()/sqrt(size), (pre, post)->0.1*(pre!=post))
    connect!(StaticSynapse, populations[:ctx_exc], populations[:ctx_inh],
             (pre, post)->25*rand()/sqrt(size), 0.1)
    connect!(StaticSynapse, populations[:ctx_inh], populations[:ctx_exc],
             (pre, post)->-25*rand()/sqrt(size), 0.4)
    connect!(StaticSynapse, populations[:ctx_inh], populations[:ctx_inh],
             (pre, post)->-25*rand()/sqrt(size), (pre, post)->0.4*(pre!=post))
    
    if ctx_str == :plastic
        connect!(EligabilitySynapse{1}, populations[:ctx_exc], populations[:str_dmsn],
                 (pre, post)->25*rand()/sqrt(size), 0.2)
        connect!(EligabilitySynapse{1}, populations[:ctx_exc], populations[:str_imsn],
                 (pre, post)->25*rand()/sqrt(size), 0.2)
    elseif ctx_str == :static
        connect!(StaticSynapse, populations[:ctx_exc], populations[:str_dmsn],
                 (pre, post)->25*rand()/sqrt(size), 0.2)
        connect!(StaticSynapse, populations[:ctx_exc], populations[:str_imsn],
                 (pre, post)->25*rand()/sqrt(size), 0.2)
    end
    if str_str == :plastic
        connect!(EligabilitySynapse{-1}, populations[:str_dmsn], populations[:str_dmsn],
             (pre, post)->-25*rand()/sqrt(size), (pre, post)->0.2*(pre!=post))
        connect!(EligabilitySynapse{-1}, populations[:str_dmsn], populations[:str_imsn],
             (pre, post)->-25*rand()/sqrt(size), 0.2)
        connect!(EligabilitySynapse{-1}, populations[:str_imsn], populations[:str_dmsn],
                 (pre, post)->-25*rand()/sqrt(size), 0.2)
        connect!(EligabilitySynapse{-1}, populations[:str_imsn], populations[:str_imsn],
                 (pre, post)->-25*rand()/sqrt(size), (pre, post)->0.2*(pre!=post))
    elseif str_str == :static
        connect!(StaticSynapse, populations[:str_dmsn], populations[:str_dmsn],
                 (pre, post)->-25*rand()/sqrt(size), (pre, post)->0.2*(pre!=post))
        connect!(StaticSynapse, populations[:str_dmsn], populations[:str_imsn],
                 (pre, post)->-25*rand()/sqrt(size), 0.2)
        connect!(StaticSynapse, populations[:str_imsn], populations[:str_dmsn],
                 (pre, post)->-25*rand()/sqrt(size), 0.2)
        connect!(StaticSynapse, populations[:str_imsn], populations[:str_imsn],
                 (pre, post)->-25*rand()/sqrt(size), (pre, post)->0.2*(pre!=post))
    end
    
    if str_snr == :plastic
        connect!(EligabilitySynapse{-1}, populations[:str_dmsn], populations[:snr],
                 (pre, post)->-5*rand()/sqrt(size), 1.0)
        connect!(EligabilitySynapse{1}, populations[:str_imsn], populations[:snr],
                 (pre, post)-> 5*rand()/sqrt(size), 1.0)
    elseif str_snr == :static
        connect!(StaticSynapse, populations[:str_dmsn], populations[:snr],
                 (pre, post)->-5*rand()/sqrt(size), 1.0)
        connect!(StaticSynapse, populations[:str_imsn], populations[:snr],
                 (pre, post)-> 5*rand()/sqrt(size), 1.0)
    end
    #connect!(StaticSynapse, populations[:snr], populations[:thal],
    #         (pre, post)->-5*rand(), 1.0)
    connect!(StaticSynapse, populations[:thal], populations[:ctx_exc],
             (pre, post)->50*rand()/Base.size(populations[:thal]), 0.2)
    connect!(StaticSynapse, populations[:thal], populations[:ctx_inh],
             (pre, post)->50*rand()/Base.size(populations[:thal]), 0.2)
    
    if str_snr == :plastic
        connect!(EligabilitySynapse{1}, populations[:thal], populations[:str_dmsn],
                 (pre, post)->30*rand()/Base.size(populations[:thal]), 0.25)
        connect!(EligabilitySynapse{1}, populations[:thal], populations[:str_imsn],
                 (pre, post)->30*rand()/Base.size(populations[:thal]), 0.25)
    elseif str_snr == :static
        connect!(StaticSynapse, populations[:thal], populations[:str_dmsn],
                 (pre, post)->30*rand()/Base.size(populations[:thal]), 0.25)
        connect!(StaticSynapse, populations[:thal], populations[:str_imsn],
                 (pre, post)->30*rand()/Base.size(populations[:thal]), 0.25)
    end
    str_dmsn_pos = rand(Base.size(populations[:str_dmsn]), 3)
    str_imsn_pos = rand(Base.size(populations[:str_imsn]), 3)
    feedback_dmsn = createFeedbackMatrix(str_dmsn_pos, readout_size, lambda=lambda, n_varicosities=n_varicosities)
    feedback_imsn = createFeedbackMatrix(str_imsn_pos, readout_size, lambda=lambda, n_varicosities=n_varicosities)
    
    balanceWeights!(populations[:ctx_exc])
    balanceWeights!(populations[:ctx_inh])
    balanceWeights!(populations[:str_dmsn])
    balanceWeights!(populations[:str_imsn])
    #ctx_exc_avg = 0.5*ones(Base.size(populations[:ctx_exc]))
    return BgNet(populations, eta_snr, eta_str, str_dmsn_pos, str_imsn_pos, feedback_dmsn, feedback_imsn, 1)
end

function train_constrained_network(desc::String; net_size=200, trial_length=200,
                        target_dim=4, target_tau=20.0, lambda=0.1,
                        n_trials=5000, striatumUpdate=:dopamine,
                        learning_rate=1e-3, feedback_factor=25,
                        n_varicosities=10,
                        ctx_str=:plastic, thal_str=:plastic, 
                        str_str=:plastic, str_snr=:plastic,
                        repetition=1)
    net = ConstrainedBgNet(net_size, target_dim, learning_rate,
                           learning_rate*feedback_factor, lambda=lambda,
                           n_varicosities=n_varicosities,
                           ctx_str=ctx_str, thal_str=thal_str, 
                           str_str=str_str, str_snr=str_snr)
    input = create_input(size(net[:thal]), trial_length)
    target = 0.5 .+ 0.15*gaussianProcessTarget(trial_length, target_dim, target_tau)
    losses = Float64[]
    @showprogress desc for trial_id=1:n_trials
        loss = run_trial(net, target, input, striatumUpdate)
        push!(losses, loss)
    end
    return net, losses
end

function run_constraint_experiment(fn; params...)
    param_ranges = NamedTuple(params)
    git_commit = readchomp(`git rev-parse --short HEAD`)
    for (i, values) in enumerate(product(param_ranges...))
        named_params = NamedTuple{keys(param_ranges)}(values)
        if named_params.striatumUpdate == :ideal
            named_params = merge(named_params, (;feedback_factor = named_params.feedback_factor*10.0))
        end
        all_params = merge(DEFAULT_PARAMS, named_params)
        net, losses = train_constrained_network("$named_params "; all_params...)
        h5open(fn, "cw") do fid
            fid["run$i"] = losses
            for k in keys(all_params)
                v = all_params[k]
                if v isa Number
                    attributes(fid["run$i"])[string(k)] = v
                else
                    attributes(fid["run$i"])[string(k)] = string(v)
                end
            end
            attributes(fid["run$i"])["git"] = git_commit
        end
    end
end
