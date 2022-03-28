include("network.jl")

mutable struct BgNet
    populations::Dict{Symbol, Population}
    eta_snr::Float64
    eta_str::Float64
    str_dmsn_pos::Matrix{Float64}
    str_imsn_pos::Matrix{Float64}
    feedback_dmsn::Matrix{Float64}
    feedback_imsn::Matrix{Float64}
    t::Int64
end

function BgNet(size::Integer, readout_size::Integer, eta_snr::Float64, eta_str::Float64;
               lambda=0.1, SynapseType::Type=EligabilitySynapse, n_varicosities=10)
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
    connect!(SynapseType{1}, populations[:ctx_exc], populations[:str_dmsn],
             (pre, post)->25*rand()/sqrt(size), 0.2)
    connect!(SynapseType{1}, populations[:ctx_exc], populations[:str_imsn],
             (pre, post)->25*rand()/sqrt(size), 0.2)
    connect!(SynapseType{-1}, populations[:str_dmsn], populations[:str_dmsn],
             (pre, post)->-25*rand()/sqrt(size), (pre, post)->0.2*(pre!=post))
    connect!(SynapseType{-1}, populations[:str_dmsn], populations[:str_imsn],
             (pre, post)->-25*rand()/sqrt(size), 0.2)
    connect!(SynapseType{-1}, populations[:str_imsn], populations[:str_dmsn],
             (pre, post)->-25*rand()/sqrt(size), 0.2)
    connect!(SynapseType{-1}, populations[:str_imsn], populations[:str_imsn],
             (pre, post)->-25*rand()/sqrt(size), (pre, post)->0.2*(pre!=post))
    connect!(SynapseType{-1}, populations[:str_dmsn], populations[:snr],
             (pre, post)->-5*rand()/sqrt(size), 1.0)
    connect!(SynapseType{1}, populations[:str_imsn], populations[:snr],
             (pre, post)-> 5*rand()/sqrt(size), 1.0)
    #connect!(StaticSynapse, populations[:snr], populations[:thal],
    #         (pre, post)->-5*rand(), 1.0)
    connect!(StaticSynapse, populations[:thal], populations[:ctx_exc],
             (pre, post)->50*rand()/Base.size(populations[:thal]), 0.2)
    connect!(StaticSynapse, populations[:thal], populations[:ctx_inh],
             (pre, post)->50*rand()/Base.size(populations[:thal]), 0.2)
    connect!(SynapseType{1}, populations[:thal], populations[:str_dmsn],
             (pre, post)->30*rand()/Base.size(populations[:thal]), 0.25)
    connect!(SynapseType{1}, populations[:thal], populations[:str_imsn],
             (pre, post)->30*rand()/Base.size(populations[:thal]), 0.25)
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

pop_order(::BgNet) = (:thal, :ctx_exc, :ctx_inh, :str_dmsn, :str_imsn, :snr)

function step!(net::BgNet; clamp::NamedTuple=NamedTuple())
    for pop in pop_order(net)
        step!(net[pop])
        if pop in keys(clamp)
            net[pop].r = clamp[pop] + net[pop].noise*randn(size(net[pop]))
        end
    end
    net.t += 1
end

function step!(net::BgNet, target; clamp::NamedTuple=NamedTuple(), updateStriatum=:dopamine)
    step!(net, clamp=clamp)
    error = net[:snr].r - target
    updateWeights!(net[:snr], -error, net.eta_snr, net.t)
    if updateStriatum==:dopamine
        postFactor = net[:snr].r .* (1 .- net[:snr].r)
        feedback_dmsn =  (net.feedback_dmsn)*(error .* postFactor)
        feedback_imsn = -(net.feedback_imsn)*(error .* postFactor)
        updateWeights!(net[:str_dmsn], feedback_dmsn, net.eta_str, net.t)
        updateWeights!(net[:str_imsn], feedback_imsn, net.eta_str, net.t)
    elseif updateStriatum==:flat_dopamine
        postFactor = net[:snr].r .* (1 .- net[:snr].r)
        feedback_dmsn =  (net.feedback_dmsn)*(error .* postFactor)
        feedback_imsn = -(net.feedback_imsn)*(error .* postFactor)
        feedback_dmsn_flat = mean(feedback_dmsn) .* ones(size(net[:str_dmsn]))
        feedback_imsn_flat = mean(feedback_imsn) .* ones(size(net[:str_imsn]))
        updateWeights!(net[:str_dmsn], feedback_dmsn_flat, net.eta_str, net.t)
        updateWeights!(net[:str_imsn], feedback_imsn_flat, net.eta_str, net.t)
    elseif updateStriatum==:ideal
        postFactor = net[:snr].r .* (1 .- net[:snr].r)
        feedback_dmsn = -(getWeightMatrix(net[:str_dmsn], net[:snr]))'*(error .* postFactor)
        feedback_imsn = -(getWeightMatrix(net[:str_imsn], net[:snr]))'*(error .* postFactor)
        updateWeights!(net[:str_dmsn], feedback_dmsn, net.eta_str, net.t)
        updateWeights!(net[:str_imsn], feedback_imsn, net.eta_str, net.t)
    end
    #net.ctx_exc_avg .= 0.999 .* net.ctx_exc_avg .+ 0.001 .* net[:ctx_exc].r
    #updateWeights!(net[:ctx_exc], 0.5 .- net.ctx_exc_avg, net.eta)
    #updateWeights!(net[:ctx_inh], 0.5 .- net[:ctx_inh].r, net.eta)
    #updateWeights!(net[:ctx_exc], nothing, net.eta)
    #updateWeights!(net[:ctx_inh], nothing, net.eta)
    #net[:snr].r .= target
    return sum(error.^2)
end

function recordSampleRun(net::BgNet, T::Integer; clamp::NamedTuple=NamedTuple())
    log = NamedTuple(pop=>zeros(size(net[pop]), T) for pop in pop_order(net))
    for t=1:T
        step!(net, clamp=map(x->x(t), clamp))
        for pop in pop_order(net)
            log[pop][:, t] = net[pop].r
        end
    end
    return log
end

function recordSampleRun(net::BgNet, T::Integer, target_fcn; clamp::NamedTuple=NamedTuple(), updateStriatum=:dopamine)
    log = NamedTuple(pop=>zeros(size(net[pop]), T) for pop in pop_order(net))
    for t=1:T
        step!(net, target_fcn(t), clamp=map(x->x(t), clamp), updateStriatum=updateStriatum)
        for pop in pop_order(net)
            log[pop][:, t] = net[pop].r
        end
    end
    return log
end

function Base.show(io::IO, net::BgNet)
    println(io, "BgNet")
    for pop in pop_order(net)
        println(io, "  $pop: $(size(net[pop])) units")
    end
    println(io, "  Learning rate: $(net.eta_snr), $(net.eta_str)")
end

Base.getindex(net::BgNet, pop::Symbol) = net.populations[pop]

function createFeedbackMatrix(str_pos, output_size; n_varicosities=10, lambda=0.1)
    str_size = size(str_pos, 1)
    DA = zeros(str_size, output_size)
    for i=1:output_size
        varicosities_pos = rand(n_varicosities, 3)
        for j=1:str_size, k=1:n_varicosities
            d = norm(str_pos[j, :] - varicosities_pos[k, :])
            DA[j, i] += (1/lambda)*exp(-d/lambda)
        end
    end
    return DA
end

## Plotting

function plotHeatmaps(log::NamedTuple)
    y = 1
    yticks = Float64[]
    p = plot()
    for pop in keys(log)
        dy, T = size(log[pop])
        heatmap!(p, 1:T, y:(y+dy-1), log[pop], clim=(0, 1))
        push!(yticks, y+0.5dy)
        y += dy+5
    end
    plot!(p, yticks=(yticks, keys(log)))
end

function plotTraces(log::NamedTuple; target=nothing, labels=nothing)
    y = 0
    yticks = Float64[]
    p = plot(size=(700, 800), fmt=:png)
    T = 0
    for pop in keys(log)
        T = size(log[pop], 2)
        if pop==:snr && target != nothing
            plot!(p, 1:T, log[pop]'.+y, color=[1 2])
            tar = target isa Function ? hcat(target.(1:T)...)' : target
            plot!(1:T, tar.+y, color=[1 2], linestyle=:dash)
        else
            plot!(p, 1:T, log[pop]'.+y)
        end
        push!(yticks, y+0.6)
        y -= 1.2
    end
    if labels==nothing
        labels = keys(log)
    end
    plot!(p, yticks=(yticks, labels), xlim=(1, T), legend=false)
end

plotHeatmaps(net::BgNet, T::Integer) = plotHeatmaps(recordSampleRun(net, T))
plotTraces(net::BgNet, T::Integer; target=nothing) = plotTraces(recordSampleRun(net, T), target=target)

function gaussianProcessTarget(duration, ndim, tau; eps=1e-6)
    dists = [min(abs(i-j), abs(duration+j-i), abs(-duration+j-i)) 
             for i=1:duration, j=1:duration]
    cov = exp.(-(dists.^2)./(tau^2))
    return rand(MvNormal(zeros(duration), cov+eps*I), ndim)
end 

function getAlignmentAngle(net::BgNet)
    fb = vcat(-net.feedback_dmsn[:], net.feedback_imsn[:])
    wm_dmsn = getWeightMatrix(net[:str_dmsn], net[:snr])'[:]
    wm_imsn = getWeightMatrix(net[:str_imsn], net[:snr])'[:]
    wm = vcat(wm_dmsn, wm_imsn)
    (fb'*wm)/norm(fb)/norm(wm)
end